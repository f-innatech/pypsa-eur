"""Add two project-sourced nuclear cost points to the toy break-even surface.

Both points use the same 85% capacity factor and the project's 2050 nuclear
operating-cost assumptions. They are compared with the toy model, not with a
PyPSA-Eur result.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_break_even_networks import (
    CHINA_AP1000_CAPEX_EUR_PER_KW,
    COMPARISON_DISCOUNT_RATE,
    THELLUFSEN_CAPEX_MEUR_PER_MW,
    THELLUFSEN_LIFETIME_YEARS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = ROOT / "results/toy_capex_opex_surface/samples.csv"
DEFAULT_COSTS = ROOT / "resources/test/costs_2050.csv"
DEFAULT_OUTPUT = (
    ROOT / "results/toy_capex_opex_surface/break_even_surface_nuclear_cf_axis_0_1.png"
)
CAPACITY_MW = 60.0
TECH_CAPACITY_FACTOR = 0.85
HOURS_PER_YEAR = 8760


def nuclear_operating_costs(path: Path) -> tuple[float, float]:
    costs = pd.read_csv(path)
    nuclear = costs.loc[costs.technology == "nuclear"].set_index("parameter")["value"]
    variable_opex = nuclear["VOM"] + nuclear["fuel"] / nuclear["efficiency"]
    fixed_om_fraction = nuclear["FOM"] / 100
    return float(variable_opex), float(fixed_om_fraction)


def plot(samples_path: Path, costs_path: Path, output: Path) -> None:
    samples = pd.read_csv(samples_path).sort_values("capacity_factor")
    known_cf = samples.capacity_factor.to_numpy()
    known_value = samples.system_value_keur_per_mw_year.to_numpy()
    if known_cf[0] > 0 or known_cf[-1] < TECH_CAPACITY_FACTOR:
        raise ValueError("Toy samples do not cover the selected capacity factor")

    opex, fixed_om_fraction = nuclear_operating_costs(costs_path)
    rate = COMPARISON_DISCOUNT_RATE
    years = THELLUFSEN_LIFETIME_YEARS
    annuity = rate / (1 - (1 + rate) ** -years)
    capex_eur_per_kw = {
        "China AP-1000": CHINA_AP1000_CAPEX_EUR_PER_KW,
        "Flamanville 3": THELLUFSEN_CAPEX_MEUR_PER_MW["Flamanville 3"] * 1000,
    }
    # EUR/kW-year and kEUR/MW-year have the same numerical value.
    points = {
        name: capex * (annuity + fixed_om_fraction)
        for name, capex in capex_eur_per_kw.items()
    }

    cf = np.linspace(0.1, 0.9, 200)
    value = np.interp(cf, known_cf, known_value)
    zero_cost_opex = np.max(value / (HOURS_PER_YEAR / 1000 * cf))
    opex_grid = np.linspace(0, zero_cost_opex * 1.1, 80)
    X, Y = np.meshgrid(opex_grid, cf)
    surface = value[:, None] - X * (HOURS_PER_YEAR / 1000 * Y)
    Z = np.where(surface >= 0, surface, np.nan)
    avoided_million = value * CAPACITY_MW / 1000
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=float(avoided_million.max()))
    color_map = plt.colormaps["viridis"]
    facecolors = color_map(color_norm(np.broadcast_to(avoided_million[:, None], Z.shape)))

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.plot_surface(
        X, Y, Z, facecolors=facecolors, shade=False, alpha=0.76, linewidth=0,
        antialiased=True, rcount=len(cf), ccount=len(opex_grid), zorder=10,
    )
    boundary = value / (HOURS_PER_YEAR / 1000 * cf)
    ax.plot(boundary, cf, 0, color="black", linewidth=1.3, zorder=3)

    surface_at_point = (
        np.interp(TECH_CAPACITY_FACTOR, known_cf, known_value)
        - HOURS_PER_YEAR / 1000 * opex * TECH_CAPACITY_FACTOR
    )
    x_axis_end = float(opex_grid[-1])
    y_axis_end = 1.0
    for name, annual_fixed_cost in points.items():
        above = annual_fixed_cost > surface_at_point
        linestyle = "-" if above else ":"
        guide_zorder = 20 if above else 3
        point_zorder = 22 if above else 5
        # Project to the three visible axis edges, without connecting to the surface.
        axis_feet = (
            (opex, 0, 0),
            (x_axis_end, TECH_CAPACITY_FACTOR, 0),
            (x_axis_end, y_axis_end, annual_fixed_cost),
        )
        for foot_x, foot_y, foot_z in axis_feet:
            ax.plot(
                [opex, foot_x], [TECH_CAPACITY_FACTOR, foot_y],
                [annual_fixed_cost, foot_z],
                color="red", linestyle=linestyle, linewidth=1.8,
                alpha=0.9, zorder=guide_zorder,
            )
            ax.scatter([foot_x], [foot_y], [foot_z], color="red", s=20,
                       depthshade=False, zorder=point_zorder)
        ax.scatter(
            [opex], [TECH_CAPACITY_FACTOR], [annual_fixed_cost],
            color="red", marker="^" if above else "o", s=105,
            edgecolor="white", linewidth=1, depthshade=False,
            zorder=point_zorder,
        )
        if not above:
            ax.plot(
                [opex, opex], [TECH_CAPACITY_FACTOR, TECH_CAPACITY_FACTOR],
                [annual_fixed_cost, surface_at_point],
                color="red", linestyle=":", linewidth=2.8, zorder=8,
            )
            ax.scatter(
                [opex], [TECH_CAPACITY_FACTOR], [surface_at_point],
                color="red", marker="x", s=65, depthshade=False, zorder=12,
            )

    ax.text(opex + 11, TECH_CAPACITY_FACTOR + 0.02, points["Flamanville 3"] + 8,
            f"Flamanville 3\n{points['Flamanville 3']:.0f} EUR/kW-year",
            color="#990000", fontsize=9, zorder=23)
    ax.text(opex + 11, TECH_CAPACITY_FACTOR - 0.09, points["China AP-1000"] - 10,
            f"China AP-1000\n{points['China AP-1000']:.0f} EUR/kW-year",
            color="#990000", fontsize=9, zorder=23)

    ax.set(
        xlim=(0, x_axis_end), ylim=(0, y_axis_end),
        zlim=(0, max(points.values()) * 1.1),
        xlabel="Variable OPEX (EUR/MWh)",
        ylabel="Capacity factor",
        zlabel="Break-even annual fixed cost (EUR/kW-year)",
        title="Toy break-even surface with two nuclear cost examples",
    )
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.view_init(elev=30, azim=-70)
    fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=ax, shrink=0.62, pad=0.12,
        label="Avoided system cost with free plant (million EUR/year)",
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="red", linewidth=2.6, label="Above break-even"),
            Line2D([0], [0], color="red", linestyle=":", linewidth=2.6,
                   label="Below break-even"),
        ],
        loc="upper right", bbox_to_anchor=(0.76, 0.88), frameon=False,
    )
    fig.text(
        0.12, 0.03,
        f"Both: {TECH_CAPACITY_FACTOR:.0%} CF, {opex:.1f} EUR/MWh OPEX. "
        f"CAPEX: project comparison script; 7% / 60 years. "
        f"Fixed O&M: {fixed_om_fraction:.2%} of CAPEX/year from project costs_2050.csv. "
        "Red guides project to the axes.",
        fontsize=8, color="0.3",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Surface at both points: {surface_at_point:.1f} EUR/kW-year")
    for name, annual_fixed_cost in points.items():
        position = "above" if annual_fixed_cost > surface_at_point else "below"
        print(f"{name}: {annual_fixed_cost:.1f} EUR/kW-year, {position}")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--costs", type=Path, default=DEFAULT_COSTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.samples, args.costs, args.output)


if __name__ == "__main__":
    main()
