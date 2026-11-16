"""Test how many PyPSA solves approximate a CAPEX/OPEX break-even surface.

Synthetic 14-day, 3-hour electricity system. No PyPSA-Eur inputs or workflow.
For a fixed free generator, force its total output E and measure the reduction
in optimized system cost. LP duals bound interpolation error between solves.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa


logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("linopy").setLevel(logging.ERROR)
HOURS_PER_YEAR = 8760
STEP_HOURS = 3
DAYS = 14
HOURS = DAYS * 24
CAPACITY_MW = 60.0
AVAILABILITY = 0.9
OUTPUT_DEFAULT = Path("results/toy_capex_opex_surface")


@dataclass(frozen=True)
class Sample:
    energy_mwh: float
    value_eur: float
    slope_eur_per_mwh: float

    @property
    def capacity_factor(self):
        return self.energy_mwh / (CAPACITY_MW * HOURS)

    @property
    def value_keur_per_mw_year(self):
        return self.value_eur * HOURS_PER_YEAR / HOURS / CAPACITY_MW / 1000


def make_network(candidate: bool) -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2050-01-01", periods=HOURS // STEP_HOURS, freq="3h"))
    n.snapshot_weightings.loc[:, ["objective", "generators", "stores"]] = STEP_HOURS
    n.add("Bus", "electricity")

    t = np.arange(len(n.snapshots))
    hour = (t * STEP_HOURS) % 24
    # Two low-wind spells and a changing demand pattern give the value curve kinks.
    wind = np.clip(
        0.48 + 0.22 * np.sin(2 * np.pi * t / 37) + 0.12 * np.sin(2 * np.pi * t / 11),
        0.08,
        0.85,
    )
    wind[25:43] *= 0.3
    wind[80:96] *= 0.4
    solar = np.maximum(0, np.sin(np.pi * (hour - 6) / 12)) * (0.8 + 0.2 * np.sin(t / 19) ** 2)
    demand = 125 + 19 * np.sin(2 * np.pi * (hour - 8) / 24) + 11 * np.sin(2 * np.pi * t / 56)

    n.add("Load", "demand", bus="electricity", p_set=demand)
    horizon_fraction = HOURS / HOURS_PER_YEAR
    n.add(
        "Generator", "wind", bus="electricity", p_nom_extendable=True,
        p_max_pu=wind, capital_cost=105_000 * horizon_fraction, marginal_cost=1,
    )
    n.add(
        "Generator", "solar", bus="electricity", p_nom_extendable=True,
        p_max_pu=solar, capital_cost=70_000 * horizon_fraction, marginal_cost=1,
    )
    n.add(
        "Generator", "backup", bus="electricity", p_nom_extendable=True,
        capital_cost=17_000 * horizon_fraction, marginal_cost=160,
    )
    n.add(
        "StorageUnit", "battery", bus="electricity", p_nom_extendable=True,
        max_hours=8, efficiency_store=0.92, efficiency_dispatch=0.92,
        cyclic_state_of_charge=True, capital_cost=95_000 * horizon_fraction,
    )
    if candidate:
        n.add(
            "Generator", "candidate", bus="electricity", p_nom=CAPACITY_MW,
            p_max_pu=AVAILABILITY, marginal_cost=0, capital_cost=0,
        )
    return n


def solve(energy_mwh: float | None = None) -> tuple[float, float | None]:
    n = make_network(candidate=energy_mwh is not None)

    def fix_energy(network, snapshots):
        dispatch = network.model["Generator-p"].loc[:, "candidate"]
        weights = network.snapshot_weightings.generators.loc[snapshots]
        network.model.add_constraints(
            (dispatch * weights).sum() == energy_mwh,
            name="candidate_energy",
        )

    status, condition = n.optimize(
        solver_name="highs",
        extra_functionality=fix_energy if energy_mwh is not None else None,
        assign_all_duals=True,
        log_to_console=False,
    )
    if (status, condition) != ("ok", "optimal"):
        raise RuntimeError(f"Solve failed at E={energy_mwh}: {status}, {condition}")
    dual = None
    if energy_mwh is not None:
        dual = float(n.model.constraints["candidate_energy"].dual.item())
    return float(n.objective), dual


def interval_error(left: Sample, right: Sample) -> tuple[float, float]:
    """Maximum gap between the chord and the two supporting tangents."""
    e0, e1 = left.energy_mwh, right.energy_mwh
    v0, v1 = left.value_eur, right.value_eur
    s0, s1 = left.slope_eur_per_mwh, right.slope_eur_per_mwh
    if e1 <= e0:
        return 0.0, e0
    if s0 < s1 - 1e-3:
        raise RuntimeError("Value slopes increased; check the energy-constraint dual sign.")
    if abs(s0 - s1) < 1e-10:
        return 0.0, (e0 + e1) / 2
    x = (v1 - s1 * e1 - v0 + s0 * e0) / (s0 - s1)
    x = float(np.clip(x, e0, e1))
    chord = v0 + (v1 - v0) * (x - e0) / (e1 - e0)
    upper = min(v0 + s0 * (x - e0), v1 + s1 * (x - e1))
    return max(0.0, upper - chord), x


def build_samples(max_runs: int, tolerance: float):
    baseline_cost, _ = solve()
    samples = {}
    history = []

    def add(energy):
        cost, dual = solve(energy)
        # The dual is dC/dE for the equality RHS; dV/dE = -dC/dE.
        sample = Sample(energy, baseline_cost - cost, -dual)
        samples[energy] = sample
        print(
            f"solve {len(samples):2d}: CF={sample.capacity_factor:.3f}, "
            f"value={sample.value_keur_per_mw_year:.2f} kEUR/MW-year, "
            f"slope={sample.slope_eur_per_mwh:.2f} EUR/MWh",
            flush=True,
        )

    add(0.0)
    add(CAPACITY_MW * AVAILABILITY * HOURS)
    while True:
        ordered = sorted(samples.values(), key=lambda s: s.energy_mwh)
        candidates = [interval_error(a, b) for a, b in zip(ordered[:-1], ordered[1:])]
        gap, energy = max(candidates, key=lambda pair: pair[0])
        normalized_gap = gap * HOURS_PER_YEAR / HOURS / CAPACITY_MW / 1000
        history.append((len(samples), normalized_gap))
        print(f"  upper bound on interpolation error: {normalized_gap:.3f} kEUR/MW-year")
        if normalized_gap <= tolerance or len(samples) >= max_runs:
            break
        if min(abs(energy - e) for e in samples) < 1e-6:
            break
        add(energy)
    return ordered, history, baseline_cost


def save_results(samples, history, baseline_cost, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "energy_mwh_14_days": [s.energy_mwh for s in samples],
            "capacity_factor": [s.capacity_factor for s in samples],
            "system_value_keur_per_mw_year": [s.value_keur_per_mw_year for s in samples],
            "value_slope_eur_per_mwh": [s.slope_eur_per_mwh for s in samples],
        }
    )
    df.to_csv(outdir / "samples.csv", index=False)
    pd.DataFrame(history, columns=["candidate_solves", "max_error_keur_per_mw_year"]).to_csv(
        outdir / "convergence.csv", index=False,
    )

    cf = np.linspace(0, AVAILABILITY, 200)
    value = np.interp(cf, df.capacity_factor, df.system_value_keur_per_mw_year)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(cf, value, label="Chord interpolation", color="tab:blue")
    axes[0].scatter(df.capacity_factor, df.system_value_keur_per_mw_year, color="black", s=25, label="Solved")
    axes[0].set(xlabel="Candidate capacity factor", ylabel="System value (kEUR/MW-year)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    run_count, gaps = zip(*history)
    axes[1].plot(run_count, gaps, marker="o")
    axes[1].set(xlabel="Candidate solves", ylabel="Maximum interpolation error\n(kEUR/MW-year)")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Synthetic PyPSA system: value and adaptive convergence")
    fig.tight_layout()
    fig.savefig(outdir / "value_and_convergence.png", dpi=180)
    plt.close(fig)

    opex_max = max(1.0, float(np.nanmax(value[1:] / (8.76 * cf[1:])))) * 1.1
    opex = np.linspace(0, opex_max, 80)
    X, Y = np.meshgrid(opex, cf)
    break_even_capex = value[:, None] - X * (8.76 * Y)  # kEUR/MW-year
    Z = np.where(break_even_capex >= 0, break_even_capex, np.nan)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, Y, Z, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True,
        rcount=len(cf), ccount=len(opex),
    )
    zero_opex = value[1:] / (8.76 * cf[1:])
    in_view = zero_opex <= opex_max
    ax.plot(zero_opex[in_view], cf[1:][in_view], 0, color="black", linewidth=1.5)
    ax.set_zlim(bottom=0)
    ax.set(
        xlabel="Variable OPEX (EUR/MWh)",
        ylabel="Capacity factor",
        zlabel="Break-even annual CAPEX (kEUR/MW-year)",
        title="Interpolated break-even surface (fixed 60 MW)",
    )
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    fig.savefig(outdir / "break_even_surface.png", dpi=180)
    plt.close(fig)

    # With unrestricted dispatch, the plant selects the best point on this
    # surface at each OPEX. The resulting envelope is the investment threshold.
    sampled_cf = df.capacity_factor.to_numpy()
    sampled_value = df.system_value_keur_per_mw_year.to_numpy()
    net_value = sampled_value[:, None] - opex[None, :] * (8.76 * sampled_cf[:, None])
    best_index = np.argmax(net_value, axis=0)
    envelope = net_value[best_index, np.arange(len(opex))]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(opex, envelope, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="Variable OPEX (EUR/MWh)",
        ylabel="Maximum affordable annual CAPEX (kEUR/MW-year)",
        title="Investment threshold with free choice of dispatch",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "investment_envelope.png", dpi=180)
    plt.close(fig)

    (outdir / "model.txt").write_text(
        f"Synthetic 14-day, 3-hour system, fixed candidate capacity {CAPACITY_MW:g} MW, "
        f"availability {AVAILABILITY:g}. Baseline horizon cost {baseline_cost:.2f} EUR.\n"
        "Value and CAPEX are scaled to annual equivalents. This is a toy-system "
        "interpolation test; its run count does not predict PyPSA-Eur's.\n"
        "The CSV's maximum error is a dual-based upper bound for the CAPEX "
        "surface at every OPEX, assuming optimal LP solutions. The investment "
        "envelope maximizes over capacity factor at each OPEX.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runs", type=int, default=30, help="Candidate solves, at least 2")
    parser.add_argument(
        "--tolerance", type=float, default=2.0,
        help="Stop at this maximum break-even CAPEX error, kEUR/MW-year",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    if args.max_runs < 2 or args.tolerance < 0:
        parser.error("--max-runs must be >= 2 and --tolerance must be >= 0")
    samples, history, baseline_cost = build_samples(args.max_runs, args.tolerance)
    save_results(samples, history, baseline_cost, args.output_dir)
    print(f"Saved {len(samples)} candidate solves and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
