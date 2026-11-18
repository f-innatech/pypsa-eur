"""Compare nine capacity-factor break-even lines with the toy model's 3D plot.

The example technology dots are invented solely to test the visual layout.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SAMPLES = Path("results/toy_capex_opex_surface/samples.csv")
DEFAULT_OUTPUT = Path("results/toy_capex_opex_surface")
CF_LEVELS = np.linspace(0.1, 0.9, 9)
HOURS_PER_YEAR = 8760


def plot_lines(samples: pd.DataFrame, output_dir: Path, example_dots: bool) -> Path:
    data = samples.sort_values("capacity_factor")
    cf = data["capacity_factor"].to_numpy()
    value = data["system_value_keur_per_mw_year"].to_numpy()
    if cf[0] > CF_LEVELS[0] or cf[-1] < CF_LEVELS[-1] - 1e-9:
        raise ValueError("Samples must cover capacity factors 10% through 90%")

    # Numerically, kEUR/MW-year equals EUR/kW-year.
    line_values = np.interp(CF_LEVELS, cf, value)
    opex = np.linspace(0, 190, 500)
    envelope = np.maximum(
        0,
        (value[:, None] - opex[None, :] * (HOURS_PER_YEAR / 1000) * cf[:, None]).max(axis=0),
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.fill_between(
        opex, 0, envelope, color="0.93",
        label="Cost-saving for at least one CF",
    )
    colors = plt.colormaps["viridis"](np.linspace(0.08, 0.88, len(CF_LEVELS)))
    for capacity_factor, system_value, color in zip(CF_LEVELS, line_values, colors):
        zero_capex_opex = system_value / ((HOURS_PER_YEAR / 1000) * capacity_factor)
        line_x = np.linspace(0, min(190, zero_capex_opex), 100)
        line_y = system_value - (HOURS_PER_YEAR / 1000) * capacity_factor * line_x
        ax.plot(line_x, line_y, color=color, linewidth=1.7, alpha=0.95)
        ax.annotate(
            f"{capacity_factor:.0%}", (zero_capex_opex, 0),
            xytext=(0, 5), textcoords="offset points", ha="center",
            color=color, fontsize=8, fontweight="bold",
        )

    ax.plot(opex, envelope, color="black", linewidth=2.7,
            label="Best capacity factor at each OPEX")

    if example_dots:
        examples = [
            ("Example nuclear, 80% CF", 30, 200, colors[7], "o", (11, -25)),
            ("Example gas CCS, 40% CF", 90, 120, colors[3], "s", (10, 12)),
        ]
        for label, cost, annual_capex, color, marker, offset in examples:
            ax.scatter(cost, annual_capex, s=125, marker=marker, color=color,
                       edgecolor="black", linewidth=1.3, zorder=6)
            ax.annotate(label, (cost, annual_capex), xytext=offset,
                        textcoords="offset points", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.text(
            0.02, 0.035,
            "Example dots are invented; compare each with its matching CF line",
            transform=ax.transAxes, fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    ax.set(
        xlim=(0, 190), ylim=(0, max(value) * 1.08),
        xlabel="Variable OPEX (EUR/MWh)",
        ylabel="Break-even annual CAPEX + fixed O&M (EUR/kW-year)",
        title="Toy model: break-even cost lines for fixed 60 MW capacity",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        "cf_break_even_lines_example_dots.png" if example_dots
        else "cf_break_even_lines.png"
    )
    output = output_dir / filename
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    samples = pd.read_csv(args.samples)
    print(plot_lines(samples, args.output_dir, example_dots=False))
    print(plot_lines(samples, args.output_dir, example_dots=True))


if __name__ == "__main__":
    main()
