"""Plot break-even CAPEX/OPEX lines for every solved baseload network.

Run with the pypsa-eur Python environment. The 0 GW network is the reference.
For each other network, the line joins:
  (0, annual cost saving / installed capacity)
  (annual cost saving / annual electricity generation, 0)

The line uses generation from the zero-OPEX solution. It is a fixed-dispatch
extrapolation, not a reoptimized result for positive OPEX.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from netCDF4 import Dataset
import numpy as np


NETWORK_DIR = Path(__file__).resolve().parent.parent / "networks"
OUTPUT_DIR = Path(__file__).resolve().parent

# Thellufsen et al., Applied Energy 371 (2024), Table 3, PDF p. 5:
# investment costs, fixed O&M, fuel, and 60-year lifetime. The paper uses 3%
# interest (PDF p. 5, Section 2.3, paragraph beginning "In the study,
# EnergyPLAN"). Reannualize its upfront costs at this model's 7% rate for
# comparison with the network lines, which are annual system-cost savings.
# The appended corrigendum, PDF p. 15, paragraph beginning "The authors
# regret", adds the missing 15 EUR/MWh variable O&M for every case.
THELLUFSEN_CAPEX_MEUR_PER_MW = {
    "Olkiluoto 3": 6.88,
    "Flamanville 3": 7.94,
    "Hinkley Point C": 8.98,
    "2050 IEA cost": 4.50,
    "2035 average": 6.18,
}
THELLUFSEN_OPEX_EUR_PER_MWH = 14.26 + 15.00 + 9.33
COMPARISON_DISCOUNT_RATE = 0.07
THELLUFSEN_LIFETIME_YEARS = 60
# OECD/NEA (2015), Table 25, printed p. 156 (PDF p. 158): Chinese
# AP-1000 investment cost at 10% is USD 2,802/kW. Applying the user's
# 2010-to-2020 inflation/exchange assumptions gives EUR 2,993/kW, rounded.
# Table 25 also gives AP-1000 O&M of USD 9.28/MWh and fuel of USD 9.33/MWh.
CHINA_AP1000_CAPEX_EUR_PER_KW = 3000.0
CHINA_AP1000_OPEX_EUR_PER_KWH = (9.28 + 9.33) * 1.1869 * 0.9 / 1000
# Liu et al., Nature 643 (2025), Fig. 1 and supplementary p. 4:
# Vogtle 3 and 4 overnight cost is USD 15/W in 2020 dollars.
VOGTLE_OVERNIGHT_CAPEX_EUR_PER_KW = 15.0 * 1000 * 0.9

# Weidlich et al. (2026), Methods, PDF p. 5: gas-CCS CAPEX, fixed OPEX,
# and variable OPEX including fuel. The underlying Fraunhofer ISI report,
# Section 1.3, printed p. 6, assumes a 40-year lifetime; reannualize its
# CAPEX at this model's 7% rather than the report's 2% discount rate.
# The paper does not state the euro price base for these gas-CCS values.
GAS_CCS_CAPEX_EUR_PER_KW = 1250.0
GAS_CCS_FIXED_OPEX_EUR_PER_KW_YEAR = 20.0
GAS_CCS_VARIABLE_OPEX_EUR_PER_MWH = 53.0
GAS_CCS_LIFETIME_YEARS = 40


@dataclass(frozen=True)
class NetworkResult:
    gw: int
    annual_cost_eur: float
    objective_constant_eur: float
    capacity_mw: float
    generation_mwh: float


def read_network(path: Path, gw: int) -> NetworkResult:
    with Dataset(path) as network:
        annual_cost_eur = float(network.getncattr("network_objective"))
        objective_constant_eur = float(network.getncattr("network_objective_constant"))

        carrier = np.asarray(network.variables["generators_carrier"][:])
        selected = carrier == "baseload_candidate"
        capacity_mw = float(np.sum(network.variables["generators_p_nom_opt"][:][selected]))

        if gw == 0:
            generation_mwh = 0.0
        else:
            capital_cost = np.asarray(network.variables["generators_capital_cost"][:][selected])
            marginal_cost = np.asarray(network.variables["generators_marginal_cost"][:][selected])
            if np.any(np.abs(capital_cost) > 1e-8) or np.any(np.abs(marginal_cost) > 1e-8):
                raise ValueError(f"Baseload plant costs are not zero in {path.name}")

            dispatch_names = network.variables["generators_t_p_i"][:]
            columns = np.asarray(
                [i for i, name in enumerate(dispatch_names) if str(name).endswith("baseload_candidate")],
                dtype=int,
            )
            if len(columns) == 0:
                raise ValueError(f"No baseload dispatch found in {path.name}")
            dispatch_mw = network.variables["generators_t_p"][:, columns]
            hours = network.variables["snapshots_generators"][:]
            generation_mwh = float(np.sum(dispatch_mw * hours[:, None]))

    if not np.isclose(capacity_mw, gw * 1000, atol=0.01):
        raise ValueError(f"Filename capacity and solved capacity disagree in {path.name}")
    return NetworkResult(gw, annual_cost_eur, objective_constant_eur, capacity_mw, generation_mwh)


def load_results() -> list[NetworkResult]:
    files = list(NETWORK_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {NETWORK_DIR}")

    indexed = []
    for path in files:
        match = re.search(r"_b(\d+)\.nc$", path.name)
        if match is None:
            raise ValueError(f"Cannot identify baseload capacity from {path.name}")
        indexed.append((int(match.group(1)), path))
    indexed.sort()

    if indexed[0][0] != 0 or len({gw for gw, _ in indexed}) != len(indexed):
        raise ValueError("Expected one 0 GW reference and unique capacity labels")
    results = [read_network(path, gw) for gw, path in indexed]
    reference_constant = results[0].objective_constant_eur
    if any(not np.isclose(result.objective_constant_eur, reference_constant, atol=0.01)
           for result in results[1:]):
        raise ValueError("Objective constants differ; cost savings need further review")
    return results


def plot(results: list[NetworkResult]) -> None:
    reference_cost = results[0].annual_cost_eur
    cases = results[1:]
    if not cases:
        raise ValueError("No positive-capacity networks found")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,
    })
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    styles = ["-", "--", "-.", (0, (5, 2, 1, 2)), ":"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.77)
    fig.text(0.12, 0.94, "Break-even cost vs 0 GW (2050) at 0.9 Capacity Factor", fontsize=13, weight="semibold")
    fig.text(0.12, 0.89, "Lines use zero-OPEX dispatch; nuclear: 7%/60 years; gas+CCS: 7%/40 years",
             fontsize=9, color="#536174")

    print("capacity_GW,saving_EUR_per_year,capacity_MW,generation_MWh,capex_EUR_per_kW_year,opex_EUR_per_kWh")
    for index, result in enumerate(cases):
        savings = reference_cost - result.annual_cost_eur
        if savings <= 0 or result.generation_mwh <= 0:
            raise ValueError(f"Invalid saving or generation for {result.gw} GW")
        capex_intercept = savings / (result.capacity_mw * 1000)
        opex_intercept = savings / (result.generation_mwh * 1000)
        print(f"{result.gw},{savings:.2f},{result.capacity_mw:.3f},{result.generation_mwh:.3f},"
              f"{capex_intercept:.8f},{opex_intercept:.10f}")
        ax.plot(
            [0, opex_intercept], [capex_intercept, 0],
            label=f"{result.gw} GW",
            color=colors[index % len(colors)],
            linestyle=styles[index % len(styles)],
            linewidth=2.2,
            marker=markers[index % len(markers)],
            markersize=5,
            clip_on=False,
            zorder=3 + len(cases) - index,
        )

    # Paper points are annualized to match the y-axis of the network lines.
    annuity = COMPARISON_DISCOUNT_RATE / (
        1 - (1 + COMPARISON_DISCOUNT_RATE) ** (-THELLUFSEN_LIFETIME_YEARS)
    )
    paper_opex_eur_per_kwh = THELLUFSEN_OPEX_EUR_PER_MWH / 1000
    paper_points = {
        name: investment_meur_per_mw * 1000 * annuity
        for name, investment_meur_per_mw in THELLUFSEN_CAPEX_MEUR_PER_MW.items()
    }
    ax.axvline(paper_opex_eur_per_kwh, color="#AAB4BE", linewidth=0.9,
               linestyle=(0, (2, 3)), zorder=1)
    ax.scatter(
        [paper_opex_eur_per_kwh] * len(paper_points),
        list(paper_points.values()),
        s=70, facecolor="white", edgecolor="#172334", linewidth=1.6, zorder=12,
    )
    for name, annualized_capex in paper_points.items():
        ax.annotate(
            name,
            (paper_opex_eur_per_kwh, annualized_capex),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#172334",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
            zorder=13,
        )
    print(f"Thellufsen_total_opex_EUR_per_kWh,{paper_opex_eur_per_kwh:.5f}")
    for name, annualized_capex in paper_points.items():
        print(f"Thellufsen_{name.replace(' ', '_')}_annualized_capex_EUR_per_kW_year,"
              f"{annualized_capex:.5f}")

    # Use the NEA AP-1000 O&M plus fuel cost for the Chinese point.
    china_annualized_capex = CHINA_AP1000_CAPEX_EUR_PER_KW * annuity
    ax.scatter(
        [CHINA_AP1000_OPEX_EUR_PER_KWH], [china_annualized_capex],
        s=85, marker="s", facecolor="#5D3A9B", edgecolor="white",
        linewidth=1.2, zorder=14,
    )
    ax.annotate(
        "China AP-1000",
        (CHINA_AP1000_OPEX_EUR_PER_KWH, china_annualized_capex),
        xytext=(8, 0), textcoords="offset points", ha="left", va="center",
        fontsize=8.5, color="#5D3A9B",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        zorder=15,
    )
    print(f"China_AP1000_annualized_capex_EUR_per_kW_year,{china_annualized_capex:.5f}")
    print(f"China_AP1000_opex_EUR_per_kWh,{CHINA_AP1000_OPEX_EUR_PER_KWH:.8f}")

    vogtle_annualized_capex = VOGTLE_OVERNIGHT_CAPEX_EUR_PER_KW * annuity
    ax.scatter(
        [paper_opex_eur_per_kwh], [vogtle_annualized_capex],
        s=95, marker="D", facecolor="#8B1E3F", edgecolor="white",
        linewidth=1.2, zorder=14,
    )
    ax.annotate(
        "Vogtle 3 & 4",
        (paper_opex_eur_per_kwh, vogtle_annualized_capex),
        xytext=(8, 0), textcoords="offset points", ha="left", va="center",
        fontsize=8.5, color="#8B1E3F",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        zorder=15,
    )
    print(f"Vogtle_3_4_annualized_capex_EUR_per_kW_year,{vogtle_annualized_capex:.5f}")

    gas_ccs_annuity = COMPARISON_DISCOUNT_RATE / (
        1 - (1 + COMPARISON_DISCOUNT_RATE) ** (-GAS_CCS_LIFETIME_YEARS)
    )
    gas_ccs_opex_eur_per_kwh = GAS_CCS_VARIABLE_OPEX_EUR_PER_MWH / 1000
    gas_ccs_annual_fixed_eur_per_kw_year = (
        GAS_CCS_CAPEX_EUR_PER_KW * gas_ccs_annuity
        + GAS_CCS_FIXED_OPEX_EUR_PER_KW_YEAR
    )
    ax.scatter(
        [gas_ccs_opex_eur_per_kwh], [gas_ccs_annual_fixed_eur_per_kw_year],
        s=105, marker="P", facecolor="#795548", edgecolor="white",
        linewidth=1.2, zorder=14,
    )
    ax.annotate(
        "Gas + CCS",
        (gas_ccs_opex_eur_per_kwh, gas_ccs_annual_fixed_eur_per_kw_year),
        xytext=(8, -10), textcoords="offset points", ha="left", va="center",
        fontsize=8.5, color="#795548",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.2},
        zorder=15,
    )
    print(f"Gas_CCS_opex_EUR_per_kWh,{gas_ccs_opex_eur_per_kwh:.8f}")
    print(f"Gas_CCS_annual_fixed_EUR_per_kW_year,{gas_ccs_annual_fixed_eur_per_kw_year:.5f}")

    ax.set_xlim(0, 0.08)
    ax.set_ylim(0, 1100)
    ax.set_xlabel("Operating cost (€/kWh)", labelpad=8)
    ax.set_ylabel("Break-even annual fixed cost (2020€/kW/year)", labelpad=9)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.grid(axis="y", color="#DCE3E9", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#8292A1")
    ax.tick_params(axis="both", colors="#526172", length=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=5,
              frameon=False, fontsize=9, handlelength=2.6, columnspacing=1.8)
    fig.text(0.12, 0.068,
             "Sources: Thellufsen et al. (2024) + corrigendum; OECD/NEA (2015), Table 25; Liu et al. (2025), Fig. 1.",
             fontsize=7.5, color="#536174")
    fig.text(0.12, 0.047,
             "China: 10% construction interest cost €3,000/kW; OPEX €0.01988/kWh. Vogtle: 2020 overnight €13,500/kW.",
             fontsize=7.5, color="#536174")
    fig.text(0.12, 0.026,
             "Gas+CCS: Weidlich et al. (2026), p. 5; €53/MWh incl. fuel; euro price base unstated; CO2 transport excluded.",
             fontsize=7.5, color="#536174")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "break_even_all_networks_with_weidlich_gas_ccs.pdf")
    fig.savefig(OUTPUT_DIR / "break_even_all_networks_with_weidlich_gas_ccs.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    plot(load_results())
