"""Show the toy break-even surface as a heatmap and filled contours."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SAMPLES = Path("results/toy_capex_opex_surface/samples.csv")
DEFAULT_OUTPUT = Path("results/toy_capex_opex_surface/break_even_heatmap.png")
HOURS_PER_YEAR = 8760


def plot_heatmap(samples: pd.DataFrame, output: Path) -> None:
    data = samples.sort_values("capacity_factor")
    known_cf = data["capacity_factor"].to_numpy()
    known_value = data["system_value_keur_per_mw_year"].to_numpy()
    if known_cf[0] > 0.1 or known_cf[-1] < 0.9 - 1e-9:
        raise ValueError("Samples must cover capacity factors from 10% to 90%")

    opex = np.linspace(0, 190, 241)
    cf = np.linspace(0.1, 0.9, 161)
    value = np.interp(cf, known_cf, known_value)
    # kEUR/MW-year has the same numerical value as EUR/kW-year.
    annual_fixed_cost = value[:, None] - (HOURS_PER_YEAR / 1000) * cf[:, None] * opex[None, :]
    visible_cost = np.ma.masked_less(annual_fixed_cost, 0)
    levels = np.arange(0, 451, 25)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("white")
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True, sharey=True,
                             layout="constrained")
    mesh = axes[0].pcolormesh(opex, cf, visible_cost, shading="auto", cmap=cmap, norm=norm)
    axes[1].contourf(opex, cf, visible_cost, levels=levels, cmap=cmap, norm=norm)
    for ax, title in zip(axes, ("Pseudocolor mesh", "Filled contours")):
        zero = ax.contour(opex, cf, annual_fixed_cost, levels=[0],
                          colors="black", linewidths=1.3)
        ax.clabel(zero, fmt={0: "zero fixed cost"}, fontsize=8, inline=True)
        ax.set(title=title, ylabel="Baseload capacity factor", ylim=(0.1, 0.9))
        ax.grid(alpha=0.12)
    axes[1].set_xlabel("Variable OPEX (EUR/MWh)")
    bar = fig.colorbar(mesh, ax=axes, ticks=np.arange(0, 451, 50),
                       shrink=0.9, pad=0.025)
    bar.set_label("Maximum annual CAPEX + fixed O&M (EUR/kW-year)")
    fig.suptitle("Toy model, fixed 60 MW: break-even costs by OPEX and capacity factor")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_heatmap(pd.read_csv(args.samples), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
