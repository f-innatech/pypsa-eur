"""Plot the 2010 PyPSA-Eur 300 GW CHP break-even surface from solved run metrics.

Each run fixes gross annual baseload generation at its labelled capacity factor.
The zero-cost system saving is measured against the common 0 GW baseline.
Between solved capacity factors, values are linearly interpolated. The surface
shows affordable annual fixed cost after subtracting variable operating cost;
it is a conditional break-even surface, not a reoptimized investment boundary.
"""

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_break_even_networks import (
    CHINA_AP1000_CAPEX_EUR_PER_KW,
    COMPARISON_DISCOUNT_RATE,
    GAS_CCS_CAPEX_EUR_PER_KW,
    GAS_CCS_FIXED_OPEX_EUR_PER_KW_YEAR,
    GAS_CCS_LIFETIME_YEARS,
    GAS_CCS_VARIABLE_OPEX_EUR_PER_MWH,
    THELLUFSEN_CAPEX_MEUR_PER_MW,
    THELLUFSEN_LIFETIME_YEARS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results/baseload-chp-energy-curve-2010"
DEFAULT_COSTS = ROOT / "resources/test/costs_2050.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "break_even_surface_technologies.png"
NUCLEAR_CAPACITY_FACTOR = 0.85
GAS_CCS_CAPACITY_FACTOR = 0.40
HOURS_PER_YEAR = 8760
EXPECTED_CF = set(range(10, 100, 10))


def load_samples(results_dir: Path) -> tuple[pd.DataFrame, float, list[int]]:
    rows = []
    for path in sorted((results_dir / "metrics").glob("*b300_cf*.csv")):
        match = re.search(r"_cf(\d+)\.csv$", path.name)
        if match is None:
            continue
        network_path = results_dir / "networks" / path.with_suffix(".nc").name
        if not network_path.is_file():
            raise FileNotFoundError(f"Metrics exist but solved network is missing: {network_path}")
        row = pd.read_csv(path).squeeze()
        if int(match.group(1)) != round(100 * float(row["target_capacity_factor"])):
            raise ValueError(f"Capacity-factor label disagrees with metrics: {path.name}")
        if abs(float(row["actual_capacity_factor"]) - float(row["target_capacity_factor"])) > 1e-5:
            raise ValueError(f"Actual capacity factor misses target: {path.name}")
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No completed 300 GW energy-curve runs in {results_dir}")

    samples = pd.DataFrame(rows).sort_values("target_capacity_factor")
    if samples.target_capacity_factor.duplicated().any():
        raise ValueError("Duplicate capacity-factor samples")
    capacity_gw = samples.baseload_gw.to_numpy(dtype=float)
    if not np.allclose(capacity_gw, capacity_gw[0]) or not np.isclose(capacity_gw[0], 300):
        raise ValueError("Expected fixed 300 GW capacity in every run")
    baseline = samples.system_cost_eur_per_year + samples.system_value_eur_per_year
    if not np.allclose(baseline, baseline.iloc[0], rtol=1e-8):
        raise ValueError("Run metrics do not share the same 0 GW baseline")
    missing = sorted(EXPECTED_CF - set(round(100 * x) for x in samples.target_capacity_factor))
    return samples, float(capacity_gw[0]), missing


def nuclear_operating_costs(path: Path) -> tuple[float, float]:
    costs = pd.read_csv(path)
    nuclear = costs.loc[costs.technology == "nuclear"].set_index("parameter")["value"]
    variable_opex = nuclear["VOM"] + nuclear["fuel"] / nuclear["efficiency"]
    fixed_om_fraction = nuclear["FOM"] / 100
    return float(variable_opex), float(fixed_om_fraction)


def plot(results_dir: Path, costs_path: Path, output: Path) -> None:
    samples, capacity_gw, missing = load_samples(results_dir)
    known_cf = samples.target_capacity_factor.to_numpy(dtype=float)
    # EUR/year divided by kW gives EUR/kW-year.
    known_value = samples.system_value_eur_per_year.to_numpy(dtype=float) / (capacity_gw * 1e6)
    if known_cf[0] > 0.1 or known_cf[-1] < 0.9:
        raise ValueError("Completed runs must include the 10% and 90% endpoints")

    opex, fixed_om_fraction = nuclear_operating_costs(costs_path)
    rate = COMPARISON_DISCOUNT_RATE
    years = THELLUFSEN_LIFETIME_YEARS
    annuity = rate / (1 - (1 + rate) ** -years)
    capex_eur_per_kw = {
        "China AP-1000": CHINA_AP1000_CAPEX_EUR_PER_KW,
        "Flamanville 3": THELLUFSEN_CAPEX_MEUR_PER_MW["Flamanville 3"] * 1000,
    }
    fixed_costs = {
        name: capex * (annuity + fixed_om_fraction)
        for name, capex in capex_eur_per_kw.items()
    }
    gas_annuity = rate / (1 - (1 + rate) ** -GAS_CCS_LIFETIME_YEARS)
    fixed_costs["Gas-CCS"] = (
        GAS_CCS_CAPEX_EUR_PER_KW * gas_annuity
        + GAS_CCS_FIXED_OPEX_EUR_PER_KW_YEAR
    )
    variable_costs = {name: opex for name in capex_eur_per_kw}
    variable_costs["Gas-CCS"] = GAS_CCS_VARIABLE_OPEX_EUR_PER_MWH
    capacity_factors = {name: NUCLEAR_CAPACITY_FACTOR for name in capex_eur_per_kw}
    capacity_factors["Gas-CCS"] = GAS_CCS_CAPACITY_FACTOR

    cf = np.linspace(0.1, 0.9, 201)
    value = np.interp(cf, known_cf, known_value)
    opex_limit = float(np.max(value / (HOURS_PER_YEAR / 1000 * cf)))
    opex_grid = np.linspace(0, opex_limit * 1.1, 90)
    X, Y = np.meshgrid(opex_grid, cf)
    surface = value[:, None] - X * (HOURS_PER_YEAR / 1000 * Y)
    Z = np.where(surface >= 0, surface, np.nan)

    saved_billion = value * capacity_gw / 1000
    norm = matplotlib.colors.Normalize(vmin=0, vmax=float(saved_billion.max()))
    cmap = plt.colormaps["viridis"]
    facecolors = cmap(norm(np.broadcast_to(saved_billion[:, None], Z.shape)))

    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(
        X, Y, Z, facecolors=facecolors, shade=False, alpha=0.76,
        linewidth=0, antialiased=True, rcount=len(cf), ccount=len(opex_grid),
        zorder=10,
    )
    ax.plot(value / (HOURS_PER_YEAR / 1000 * cf), cf, 0,
            color="black", linewidth=1.3, zorder=3)
    ax.scatter(np.zeros(len(known_cf)), known_cf, known_value,
               color="black", s=17, depthshade=False, zorder=12)

    x_axis_end = float(opex_grid[-1])
    for name, annual_fixed_cost in fixed_costs.items():
        tech_opex = variable_costs[name]
        tech_cf = capacity_factors[name]
        surface_at_point = (
            np.interp(tech_cf, known_cf, known_value)
            - HOURS_PER_YEAR / 1000 * tech_opex * tech_cf
        )
        above = annual_fixed_cost > surface_at_point
        linestyle = "-" if above else ":"
        guide_zorder = 20 if above else 3
        point_zorder = 22 if above else 5
        axis_feet = (
            (tech_opex, 0, 0),
            (x_axis_end, tech_cf, 0),
            (x_axis_end, 1.0, annual_fixed_cost),
        )
        for foot_x, foot_y, foot_z in axis_feet:
            ax.plot([tech_opex, foot_x], [tech_cf, foot_y],
                    [annual_fixed_cost, foot_z], color="red", linestyle=linestyle,
                    linewidth=1.8, alpha=0.9, zorder=guide_zorder)
            ax.scatter([foot_x], [foot_y], [foot_z], color="red", s=20,
                       depthshade=False, zorder=point_zorder)
        marker = "s" if name == "Gas-CCS" else ("^" if above else "o")
        ax.scatter([tech_opex], [tech_cf], [annual_fixed_cost],
                   color="red", marker=marker, s=105,
                   edgecolor="white", linewidth=1, depthshade=False,
                   zorder=point_zorder)
        if not above:
            ax.plot([tech_opex, tech_opex], [tech_cf, tech_cf],
                    [annual_fixed_cost, surface_at_point], color="red",
                    linestyle=":", linewidth=2.8, zorder=8)
            ax.scatter([tech_opex], [tech_cf], [surface_at_point],
                       color="red", marker="x", s=65, depthshade=False, zorder=12)

    ax.text(opex + 11, NUCLEAR_CAPACITY_FACTOR + 0.02,
            fixed_costs["Flamanville 3"] + 8,
            f"Flamanville 3\n{fixed_costs['Flamanville 3']:.0f} EUR/kW-year",
            color="#990000", fontsize=9, zorder=23)
    ax.text(opex + 11, NUCLEAR_CAPACITY_FACTOR - 0.09,
            fixed_costs["China AP-1000"] - 10,
            f"China AP-1000\n{fixed_costs['China AP-1000']:.0f} EUR/kW-year",
            color="#990000", fontsize=9, zorder=23)
    ax.text(GAS_CCS_VARIABLE_OPEX_EUR_PER_MWH + 12, GAS_CCS_CAPACITY_FACTOR - 0.1,
            fixed_costs["Gas-CCS"] - 20,
            f"Gas-CCS ({GAS_CCS_CAPACITY_FACTOR:.0%} CF)\n"
            f"{fixed_costs['Gas-CCS']:.0f} EUR/kW-year",
            color="#990000", fontsize=9, zorder=23)

    ax.set(
        xlim=(0, x_axis_end), ylim=(0, 1),
        zlim=(0, max(fixed_costs.values()) * 1.1),
        xlabel="Variable OPEX (EUR/MWh)",
        ylabel="Capacity factor",
        zlabel="Break-even annual fixed cost (EUR/kW-year)",
        title=f"2010 weather, 2050 zero CO2: break-even surface (fixed {capacity_gw:g} GW CHP)",
    )
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.view_init(elev=30, azim=-70)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, shrink=0.62, pad=0.12,
                 label="Avoided system cost with free plant (billion EUR/year)")
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
               markeredgecolor="black", label="Solved CF", markersize=5),
        Line2D([0], [0], color="red", linewidth=2.6, label="Above break-even"),
        Line2D([0], [0], color="red", linestyle=":", linewidth=2.6,
               label="Below break-even"),
    ], loc="upper right", bbox_to_anchor=(0.78, 0.88), frameon=False)
    fig.text(0.08, 0.038,
             f"Nuclear: {NUCLEAR_CAPACITY_FACTOR:.0%} CF, {opex:.1f} EUR/MWh, 60 years; "
             f"Gas-CCS: {GAS_CCS_CAPACITY_FACTOR:.0%} CF, 53 EUR/MWh, 40 years. "
             "Annualization: 7%.",
             fontsize=8, color="0.3")
    note = "Gas comparison credits zero-emission CHP; residual CO2 and CO2-network costs excluded."
    if missing:
        note += f" Missing {', '.join(f'{x}%' for x in missing)} run; linearly interpolated."
    fig.text(0.08, 0.018, note, fontsize=8, color="0.3")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Solved CFs: {', '.join(f'{x:.0%}' for x in known_cf)}")
    if missing:
        print(f"Missing CFs (interpolated): {', '.join(f'{x}%' for x in missing)}")
    for name, annual_fixed_cost in fixed_costs.items():
        tech_cf = capacity_factors[name]
        surface_at_point = (
            np.interp(tech_cf, known_cf, known_value)
            - HOURS_PER_YEAR / 1000 * variable_costs[name] * tech_cf
        )
        position = "above" if annual_fixed_cost > surface_at_point else "below"
        print(f"{name} at {tech_cf:.0%} CF: fixed {annual_fixed_cost:.1f}, "
              f"break-even {surface_at_point:.1f} EUR/kW-year, {position}")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--costs", type=Path, default=DEFAULT_COSTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.results_dir, args.costs, args.output)


if __name__ == "__main__":
    main()
