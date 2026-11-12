"""Compare endogenous CAPEX/OPEX baseload cases and the 0 GW reference.

Uses the same extraction methods as compare_baseload_networks.py, but labels
cases by their CAPEX/OPEX split and reports their optimized baseload capacity.

Run from the repository root, for example:
    python "results/Plots Scripts/compare_baseload_cost_cases.py" \
        results/opex-capex-test-2010 \
        --baseline results/baseload-chp-free-opex_2010/networks/base_s_128___2050_b0.nc
"""

import argparse
import gc
import importlib.util
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa


SOURCE = Path(__file__).with_name("compare_baseload_networks.py")
CASE_RE = re.compile(r"_cost_capex(?P<percent>\d+)\.nc$")
BASELINE_LABEL = "0 GW"
PALETTE = plt.get_cmap("tab20")


def load_extractors():
    spec = importlib.util.spec_from_file_location("baseload_capacity_comparison", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_cases(inputs):
    paths = []
    for item in inputs:
        if item.is_file():
            paths.append(item)
        elif item.is_dir():
            paths.extend(item.rglob("*.nc"))
        else:
            raise FileNotFoundError(item)
    cases = {}
    for path in paths:
        match = CASE_RE.search(path.name)
        if match is None:
            continue
        percent = int(match["percent"])
        if not 0 < percent < 100:
            raise ValueError(f"Invalid CAPEX share in {path.name}")
        if percent in cases:
            raise ValueError(f"Duplicate {percent}% CAPEX case: {path}")
        cases[percent] = path.resolve()
    if not cases:
        raise ValueError("No *_cost_capexXX.nc networks found")
    return [(f"{percent}/{100 - percent}", percent, cases[percent])
            for percent in sorted(cases, reverse=True)]


def candidate_summary(n, legacy, case, percent, path):
    names = n.generators.index[n.generators.carrier == "baseload_candidate"]
    plants = n.generators.loc[names]
    capacities = legacy.nominal_capacity(plants, "p_nom", "p_nom_opt")
    built_gw = float(capacities.sum() / 1e3)
    weights = legacy.snapshot_weights(n)
    generation_mwh = float(
        n.generators_t.p.reindex(index=n.snapshots, columns=names, fill_value=0)
        .mul(weights, axis=0).sum().sum()
    )
    capital_cost = pd.to_numeric(plants.capital_cost, errors="coerce")
    marginal_cost = pd.to_numeric(plants.marginal_cost, errors="coerce")
    if len(names) and (capital_cost.nunique() != 1 or marginal_cost.nunique() != 1):
        raise ValueError(f"Candidate costs differ between nodes: {path}")
    return dict(
        case=case,
        capex_share_percent=percent,
        weather_year=pd.Timestamp(n.snapshots[0]).year,
        built_gw=built_gw,
        system_cost_eur_per_year=float(n.objective),
        annualized_capex_eur_per_mw_year=(
            float(capital_cost.iloc[0]) if len(names) else np.nan
        ),
        variable_opex_eur_per_mwh=(
            float(marginal_cost.iloc[0]) if len(names) else np.nan
        ),
        baseload_generation_twh=generation_mwh / 1e6,
        candidate_annualized_capex_eur_per_year=float((capacities * capital_cost).sum()),
        candidate_variable_opex_eur_per_year=float(generation_mwh * marginal_cost.iloc[0])
        if len(names) else 0.0,
        network_file=str(path),
    )


def extract_all(cases, baseline, legacy):
    collected = {key: [] for key in (
        "cost_details", "capacities", "generation", "energy", "curtailment",
        "demand", "nodes", "heat_assets", "heat_loads", "heat_summary",
    )}
    summaries = []
    profile = None
    weather_year = None
    entries = [(BASELINE_LABEL, 0, baseline)] + cases
    for case, percent, path in entries:
        print(f"Loading {case}: {path.name}", flush=True)
        n = pypsa.Network(path)
        year = pd.Timestamp(n.snapshots[0]).year
        if weather_year is None:
            weather_year = year
        elif year != weather_year:
            raise ValueError(f"Weather years differ: {path}")
        if not math.isfinite(float(n.objective)):
            raise ValueError(f"Unsolved network: {path}")
        summary = candidate_summary(n, legacy, case, percent, path)
        if case == BASELINE_LABEL and summary["built_gw"] > 1e-6:
            raise ValueError("The baseline network contains candidate baseload capacity")
        built = summary["built_gw"]
        summaries.append(summary)

        def put(key, value):
            if value is None:
                return
            frame = pd.DataFrame(value)
            if not frame.empty:
                frame["case"] = case
                collected[key].append(frame)

        put("cost_details", legacy.system_cost_breakdown(n, year, built))
        put("capacities", legacy.installed_capacities(n, year, built))
        put("generation", legacy.electric_generation_capacities(n, year, built))
        put("energy", legacy.electricity_generation_energy(n, year, built))
        put("curtailment", legacy.renewable_curtailment(n, year, built))
        put("demand", legacy.sector_final_energy_demand(n, year, built))
        put("nodes", legacy.baseload_siting(n, year, built))
        heat_assets, heat_loads, heat_summary = legacy.district_heating(n, year, built)
        put("heat_assets", heat_assets)
        put("heat_loads", heat_loads)
        put("heat_summary", [heat_summary] if heat_summary is not None else None)
        if case == BASELINE_LABEL:
            profile = legacy.sector_load_profile(n).reset_index()
            profile["weather_year"] = year
            profile["baseload_gw"] = 0.0
        print(f"  built {built:.2f} GW; cost {n.objective / 1e9:.2f} bn EUR/year", flush=True)
        del n
        gc.collect()

    frames = {
        key: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for key, parts in collected.items()
    }
    summary = pd.DataFrame(summaries)
    base_cost = summary.loc[summary.case == BASELINE_LABEL, "system_cost_eur_per_year"].iloc[0]
    summary["saving_vs_baseline_eur_per_year"] = (
        base_cost - summary.system_cost_eur_per_year
    )
    return summary, frames, profile, weather_year


def prepare_tables(summary, frames, legacy):
    capacities = frames["capacities"]
    keys = legacy.CAPACITY_KEYS
    cases = summary[["case", "weather_year", "built_gw"]].rename(
        columns={"built_gw": "baseload_gw"}
    )
    capacity_keys = capacities[keys].drop_duplicates()
    capacities = cases.merge(capacity_keys, how="cross").merge(
        capacities.drop(columns=["weather_year", "baseload_gw"]),
        on=["case"] + keys,
        how="left",
    )
    capacities["installed_capacity"] = capacities.installed_capacity.fillna(0)
    base_capacity = capacities.loc[capacities.case == BASELINE_LABEL, keys + ["installed_capacity"]]
    base_capacity = base_capacity.rename(columns={"installed_capacity": "baseline_capacity"})
    capacities = capacities.merge(base_capacity, on=keys, how="left")
    capacities["baseline_capacity"] = capacities.baseline_capacity.fillna(0)
    capacities["delta_vs_baseline"] = (
        capacities.installed_capacity - capacities.baseline_capacity
    )
    frames["capacities"] = capacities

    generation = frames["generation"].groupby(
        ["case", "weather_year", "baseload_gw", "component", "carrier"], as_index=False
    ).installed_capacity_gw_e.sum()
    generator_keys = generation[["component", "carrier"]].drop_duplicates()
    generation = cases.merge(generator_keys, how="cross").merge(
        generation.drop(columns=["weather_year", "baseload_gw"]),
        on=["case", "component", "carrier"],
        how="left",
    )
    generation["installed_capacity_gw_e"] = generation.installed_capacity_gw_e.fillna(0)
    base = generation.loc[generation.case == BASELINE_LABEL,
                          ["component", "carrier", "installed_capacity_gw_e"]]
    base = base.rename(columns={"installed_capacity_gw_e": "baseline_capacity_gw_e"})
    generation = generation.merge(base, on=["component", "carrier"], how="left")
    generation["baseline_capacity_gw_e"] = generation.baseline_capacity_gw_e.fillna(0)
    generation["delta_vs_baseline_gw_e"] = (
        generation.installed_capacity_gw_e - generation.baseline_capacity_gw_e
    )
    frames["generation"] = generation

    frames["energy"] = frames["energy"].groupby(
        ["case", "weather_year", "baseload_gw", "component", "carrier"], as_index=False
    ).generation_twh.sum()
    nodes = frames["nodes"]
    countries = nodes.groupby(
        ["case", "weather_year", "baseload_gw", "country"], as_index=False
    )[["installed_capacity_gw", "generation_twh"]].sum()
    countries["capacity_share_percent"] = np.where(
        countries.baseload_gw > 0,
        countries.installed_capacity_gw / countries.baseload_gw.replace(0, np.nan) * 100,
        0.0,
    )
    countries["full_load_hours"] = (
        countries.generation_twh * 1e3
        / countries.installed_capacity_gw.where(countries.installed_capacity_gw > 1e-9)
    )
    frames["countries"] = countries

    assets = frames["heat_assets"]
    if not assets.empty:
        grouping = ["case", "weather_year", "baseload_gw", "component", "carrier",
                    "role", "cost_scope"]
        assets = assets.groupby(grouping, as_index=False).sum(numeric_only=True)
        assets["direct_asset_cost_bn_eur"] = (
            assets.annualized_fixed_cost_bn_eur + assets.direct_variable_cost_bn_eur
        )
        frames["heat_assets"] = assets
    heating = frames["heat_summary"]
    if not heating.empty:
        heating["dedicated_heat_asset_direct_cost_bn_eur"] = (
            heating.dedicated_heat_asset_fixed_cost_bn_eur
            + heating.dedicated_heat_asset_variable_cost_bn_eur
        )
    return frames


def save_tables(output, summary, frames, profile):
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "system_costs.csv", index=False)
    names = {
        "cost_details": "system_cost_breakdown.csv",
        "capacities": "installed_capacities.csv",
        "generation": "generation_capacities.csv",
        "energy": "electricity_generation_twh.csv",
        "curtailment": "renewable_curtailment_twh.csv",
        "demand": "sector_final_energy_demand_twh.csv",
        "nodes": "baseload_siting_nodes.csv",
        "countries": "baseload_siting_countries.csv",
        "heat_assets": "district_heating_technologies.csv",
        "heat_loads": "district_heating_loads.csv",
        "heat_summary": "district_heating_summary.csv",
    }
    for key, name in names.items():
        frames[key].to_csv(output / name, index=False)
    profile.to_csv(output / "sector_load_profile_gw.csv", index=False)
    (output / "notes.txt").write_text(
        "Case labels give the annualized CAPEX/variable OPEX share used to price "
        "the free 300 GW reference. Built GW is optimized and can differ from 300. "
        "The 0 GW reference comes from --baseline. Cost accounting, sector demand "
        "and district-heating definitions follow compare_baseload_networks.py. "
        "Existing fixed assets' capital costs are excluded from the objective. "
        "District-heating asset costs exclude upstream electricity and fuel costs.\n",
        encoding="utf-8",
    )


def ordered_pivot(frame, order, value, column="carrier", top=None):
    if frame.empty:
        return pd.DataFrame(index=order)
    values = frame.pivot_table(
        index="case", columns=column, values=value, aggfunc="sum", fill_value=0
    ).reindex(order, fill_value=0)
    ranking = values.abs().max().sort_values(ascending=False)
    if top and len(ranking) > top:
        keep = ranking.head(top).index
        other = values.drop(columns=keep).sum(axis=1)
        values = values[keep].copy()
        if other.abs().max() > 1e-6:
            values["Other"] = other
    else:
        values = values[ranking.index]
    return values


def draw_stack(ax, values, legacy, overlay=None):
    x = np.arange(len(values))
    positive = np.zeros(len(values))
    negative = np.zeros(len(values))
    totals = values.abs().sum(axis=1).to_numpy(dtype=float)
    for i, carrier in enumerate(values.columns):
        heights = values[carrier].to_numpy(dtype=float)
        bottom = np.where(heights >= 0, positive, negative)
        bars = ax.bar(x, heights, bottom=bottom, label=carrier,
                      color=PALETTE(i % PALETTE.N))
        legacy.label_stacked_segments(ax, bars, heights, bottom, totals,
                                      force=carrier == "baseload_candidate")
        positive += np.maximum(heights, 0)
        negative += np.minimum(heights, 0)
    if overlay is not None:
        ax.plot(x, overlay.reindex(values.index), "ko--", label="District heat demand")
    ax.set_xticks(x, values.index)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)


def stacked_bars(values, title, ylabel, output, legacy, overlay=None):
    if values.empty or values.shape[1] == 0:
        return
    fig, ax = plt.subplots(figsize=(max(9, len(values) * 1.5), 5.8))
    draw_stack(ax, values, legacy, overlay)
    ax.set(xlabel="CAPEX/OPEX split (baseline first)", ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(values, title, ylabel, output, order, legacy):
    if values.empty or values.shape[1] == 0:
        return
    values = values.loc[values.abs().max(axis=1).sort_values(ascending=False).index]
    values = values.loc[values.abs().max(axis=1) > 1e-6]
    if values.empty:
        return
    x = np.arange(len(values))
    width = 0.8 / len(order)
    fig, ax = plt.subplots(figsize=(max(12, 0.8 * len(values) + 4), 5.8))
    peak = max(values.max().max(), 1)
    for i, case in enumerate(order):
        numbers = values[case].to_numpy(dtype=float)
        bars = ax.bar(x + (i - (len(order) - 1) / 2) * width,
                      numbers, width=width, label=case)
        legacy.label_grouped_bars(ax, bars, numbers, max(0.04 * peak, 1))
    ax.set_ylim(0, peak * 1.22)
    ax.set_xticks(x, values.index, rotation=45, ha="right")
    ax.set(xlabel="Technology", ylabel=ylabel, title=title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="CAPEX/OPEX", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def heatmap(values, title, colorbar, output, diverging=False, annotate=False):
    if values.empty or values.shape[0] == 0:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(values.columns) * 1.4),
                                    max(4.5, len(values) * 0.35 + 1.5)))
    maximum = max(np.nanmax(np.abs(values.to_numpy(dtype=float))), 1e-9)
    if diverging:
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-maximum,
                          vmax=maximum)
    else:
        image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(range(len(values.columns)), values.columns)
    ax.set_yticks(range(len(values)), values.index)
    ax.set(xlabel="CAPEX/OPEX split (baseline first)", title=title)
    if annotate:
        for row, numbers in enumerate(values.to_numpy(dtype=float)):
            for column, number in enumerate(numbers):
                if number >= 0.05:
                    ax.text(column, row, f"{number:.1f}", ha="center", va="center",
                            color="black" if number > maximum / 2 else "white",
                            fontsize=8)
    fig.colorbar(image, ax=ax, label=colorbar)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_summary(summary, output, order):
    values = summary.set_index("case").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    quantities = [
        ("system_cost_eur_per_year", 1e9, "System cost", "Billion EUR/year"),
        ("saving_vs_baseline_eur_per_year", 1e9, "Saving vs 0 GW", "Billion EUR/year"),
        ("built_gw", 1, "Optimized baseload capacity", "GW"),
    ]
    for ax, (column, scale, title, ylabel) in zip(axes, quantities):
        numbers = values[column].to_numpy(dtype=float) / scale
        bars = ax.bar(range(len(order)), numbers)
        for bar, number in zip(bars, numbers):
            ax.annotate(f"{number:,.1f}", (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_xticks(range(len(order)), order)
        ax.set(title=title, ylabel=ylabel, xlabel="CAPEX/OPEX split")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(min(0, numbers.min() * 1.1), max(1, numbers.max() * 1.13))
    fig.tight_layout()
    fig.savefig(output / "system_costs.png", dpi=160)
    plt.close(fig)


def plot_breakdown(frames, output, order, year, legacy):
    costs = frames["cost_details"].copy()
    costs["cost_bn"] = costs.total_cost_eur_per_year / 1e9
    groups = [
        (costs, "category", "Annual system cost by function", 12),
        (costs.loc[costs.category == "Electricity generation"], "carrier",
         "Electricity-generation cost by technology", 15),
        (costs.loc[costs.category.isin(["Storage", "Infrastructure"])], "carrier",
         "Storage and infrastructure cost", 15),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for ax, (table, column, title, top) in zip(axes, groups):
        values = ordered_pivot(table, order, "cost_bn", column, top)
        if values.shape[1]:
            draw_stack(ax, values, legacy)
        ax.set(title=title, ylabel="Billion EUR/year")
    axes[-1].set(xlabel="CAPEX/OPEX split (baseline first)")
    fig.suptitle(f"{year}: annual system cost breakdown")
    fig.tight_layout(rect=(0, 0, 0.78, 0.98))
    fig.savefig(output / f"system_cost_breakdown_{year}.png", dpi=160)
    plt.close(fig)


def plot_capacity_changes(frames, output, order, year):
    capacities = frames["capacities"]
    for (component, unit, basis), group in capacities.groupby(
        ["component", "unit", "capacity_basis"]
    ):
        if component == "Store":
            group = group.loc[group.carrier.str.contains(
                r"battery|water tanks|H2 Store", case=False, regex=True
            )]
        if group.empty:
            continue
        values = ordered_pivot(group, order, "delta_vs_baseline").T
        selected = values.index[:12].tolist()
        if component == "Generator" and "baseload_candidate" in values.index:
            if "baseload_candidate" not in selected:
                selected.append("baseload_candidate")
        values = values.loc[selected]
        if values.abs().max().max() < 1e-6:
            continue
        heatmap(values, f"{year}: {component} capacity change vs 0 GW ({basis})",
                f"Change ({unit})", output / f"capacity_change_{year}_{component.lower()}.png",
                diverging=True)


def plot_links(frames, output, order, year):
    links = frames["capacities"].loc[lambda x: x.component == "Link"]
    if links.empty:
        return
    for case in order:
        group = links.loc[links.case == case]
        values = group.groupby("carrier").installed_capacity.sum()
        values = values.loc[values > 1e-3].sort_values()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, max(5, len(values) * 0.35 + 1.5)))
        colors = ["#d55e00" if "electrolys" in carrier.lower() else "#4477aa"
                  for carrier in values.index]
        bars = ax.barh(values.index, values, color=colors)
        ax.set_xlim(0, values.max() * 1.17)
        for bar, value in zip(bars, values):
            ax.text(value + values.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2, f"{value:,.1f}",
                    va="center", fontsize=8)
        ax.set(title=f"{year}: Link capacity, {case} CAPEX/OPEX",
               xlabel="Installed Link input capacity (model units / 1,000)")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"link_capacities_{year}_{case.replace('/', '-')}.png",
                    dpi=160)
        plt.close(fig)
    electrolysis = links.loc[links.carrier.str.contains("electrolys", case=False)]
    numbers = electrolysis.groupby("case").installed_capacity.sum().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(order, numbers)
    for bar, value in zip(bars, numbers):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.1f}",
                ha="center", va="bottom")
    ax.set_ylim(0, max(numbers.max(), 1) * 1.15)
    ax.set(title=f"{year}: electrolyzer capacity", ylabel="Electricity input (GW)",
           xlabel="CAPEX/OPEX split")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / f"electrolyzer_capacity_{year}.png", dpi=160)
    plt.close(fig)


def plot_generation(frames, output, order, year, legacy):
    generation = frames["generation"]
    installed = ordered_pivot(generation, order, "installed_capacity_gw_e")
    stacked_bars(installed, f"{year}: installed electricity-generation capacity",
                 "GW electrical", output / f"generation_capacity_stacked_{year}.png", legacy)
    grouped_bars(installed.T, f"{year}: generation technologies by cost case",
                 "Installed electrical capacity (GW)",
                 output / f"generation_capacity_grouped_{year}.png", order, legacy)
    energy = ordered_pivot(frames["energy"], order, "generation_twh")
    grouped_bars(energy.T, f"{year}: electricity output by technology",
                 "Annual electricity supplied (TWh)",
                 output / f"electricity_generation_{year}.png", order, legacy)


def plot_curtailment(frames, output, order, year, legacy):
    curtailed = ordered_pivot(frames["curtailment"], order, "curtailed_twh")
    stacked_bars(curtailed, f"{year}: renewable curtailment by source",
                 "Curtailed electricity (TWh/year)",
                 output / f"renewable_curtailment_{year}.png", legacy)


def plot_sector_demand(frames, output, order, year):
    demand = frames["demand"]
    sectors = demand.loc[demand.case == BASELINE_LABEL].groupby("sector").demand_twh.sum()
    sectors = sectors.reindex(["Transport", "Heating", "Industry"], fill_value=0)
    energy = frames["energy"]
    baseload = energy.loc[energy.carrier == "baseload_candidate"].groupby("case").generation_twh.sum()
    labels = list(sectors.index) + [f"Baseload {case}\n(electricity)" for case in order[1:]]
    numbers = list(sectors) + [baseload.get(case, 0) for case in order[1:]]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 5.5))
    bars = ax.bar(labels, numbers)
    ax.axvline(2.5, color="0.5", linewidth=1)
    for bar, value in zip(bars, numbers):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:,.0f}", ha="center", va="bottom")
    ax.set_ylim(0, max(numbers) * 1.14)
    ax.set(title=f"{year}: sector final-energy demand and baseload output",
           ylabel="TWh/year")
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.02, 0.01, "Sector demand includes fuels and heat; baseload output is electricity.",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / f"sector_final_energy_demand_{year}.png", dpi=160)
    plt.close(fig)


def plot_siting(frames, output, order, year):
    for name, column, limit in (("countries", "country", None),
                                ("nodes", "node", 30)):
        table = frames[name].copy()
        if table.empty:
            continue
        if name == "nodes":
            table["location"] = table.node + " (" + table.country + ")"
        else:
            table["location"] = table[column]
        values = table.pivot_table(index="location", columns="case",
                                   values="installed_capacity_gw", aggfunc="sum",
                                   fill_value=0).reindex(columns=order, fill_value=0)
        values = values.loc[values.max(axis=1).sort_values(ascending=False).index]
        values = values.loc[values.max(axis=1) >= 0.01]
        if limit:
            values = values.head(limit)
        heatmap(values, f"{year}: baseload capacity by {name}", "Installed GW",
                output / f"baseload_siting_{name}_{year}.png",
                annotate=name == "countries")


def plot_district_heating(frames, output, order, year, legacy):
    assets = frames["heat_assets"]
    summary = frames["heat_summary"]
    if assets.empty or summary.empty:
        return
    supply = assets.loc[assets.role.isin(["heat_supply", "chp_heat", "process_heat"])]
    heat_demand = summary.set_index("case").heat_at_bus_twh_th
    for table, column, title, ylabel, filename, overlay in (
        (supply, "heat_injected_twh_th", "heat injected into district heating",
         "TWh thermal/year", "district_heating_generation", heat_demand),
        (supply, "installed_heat_output_capacity_gw_th", "district-heat production capacity",
         "GW thermal", "district_heating_capacity", None),
        (assets.loc[assets.cost_scope == "district_heat_only"], "direct_asset_cost_bn_eur",
         "dedicated district-heat asset costs", "Billion EUR/year",
         "district_heating_direct_cost", None),
    ):
        values = ordered_pivot(table, order, column)
        stacked_bars(values, f"{year}: {title}", ylabel,
                     output / f"{filename}_{year}.png", legacy, overlay)
    values = summary.set_index("case").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    storage = values.thermal_storage_capacity_gwh_th.fillna(0)
    axes[0].bar(order, storage)
    axes[0].set(title="Water-tank capacity", ylabel="GWh thermal")
    axes[1].plot(order, values.storage_charge_twh_th, marker="o", label="Charging")
    axes[1].plot(order, values.storage_discharge_twh_th, marker="o", label="Discharging")
    axes[1].set(title="Storage operation", ylabel="TWh thermal/year")
    axes[1].legend()
    for ax in axes:
        ax.set(xlabel="CAPEX/OPEX split")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / f"district_heating_storage_{year}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="Cost-case .nc files or a directory containing them")
    parser.add_argument("--baseline", type=Path,
                        default=Path("results/baseload-chp-free-opex_2010/networks/"
                                     "base_s_128___2050_b0.nc"),
                        help="Matching 0 GW solved network")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/opex-capex-test-2010/comparison"))
    args = parser.parse_args()
    if not args.baseline.is_file():
        parser.error(f"Baseline network not found: {args.baseline}")
    cases = discover_cases(args.inputs)
    legacy = load_extractors()
    summary, frames, profile, year = extract_all(cases, args.baseline.resolve(), legacy)
    order = summary.case.tolist()
    frames = prepare_tables(summary, frames, legacy)
    save_tables(args.output_dir, summary, frames, profile)
    plot_summary(summary, args.output_dir, order)
    plot_breakdown(frames, args.output_dir, order, year, legacy)
    plot_capacity_changes(frames, args.output_dir, order, year)
    plot_links(frames, args.output_dir, order, year)
    plot_generation(frames, args.output_dir, order, year, legacy)
    plot_curtailment(frames, args.output_dir, order, year, legacy)
    plot_sector_demand(frames, args.output_dir, order, year)
    legacy.plot_sector_load_profiles(profile, args.output_dir)
    plot_siting(frames, args.output_dir, order, year)
    plot_district_heating(frames, args.output_dir, order, year, legacy)
    print(f"Wrote comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
