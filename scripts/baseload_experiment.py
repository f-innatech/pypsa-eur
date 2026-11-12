# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Capacity and accounting helpers for the baseload weather experiment."""

import copy
import logging
import math
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
CARRIER = "baseload_candidate"
HEAT_CARRIER = "baseload_candidate_heat"


def add_baseload_generators(
    n, capacity_gw, cost_file, cost_config, max_hours, availability_file,
    district_heat=None, capacity_mode="fixed",
    annualized_capex_eur_per_mw_year=0.0, variable_opex_eur_per_mwh=0.0,
    must_run=False,
):
    """Offer candidate capacity with nuclear-like availability at every AC bus.

    In fixed mode, ``add_baseload_capacity_constraint`` imposes an exact total.
    Endogenous mode has no total or per-generator capacity upper bound.
    Must-run mode fixes dispatch at the country availability factor in every snapshot.
    """
    capacity_gw = float(capacity_gw)
    if capacity_gw < 0:
        raise ValueError("Baseload capacity must be non-negative")
    if capacity_mode not in {"fixed", "endogenous"}:
        raise ValueError(f"Unknown baseload capacity mode: {capacity_mode}")
    if any(
        not math.isfinite(float(value)) or value < 0
        for value in (annualized_capex_eur_per_mw_year, variable_opex_eur_per_mwh)
    ):
        raise ValueError("Baseload costs must be finite and non-negative")
    if capacity_gw == 0:
        return

    from add_electricity import load_costs

    buses = n.buses.index[n.buses.carrier == "AC"]
    if buses.empty:
        raise ValueError("No AC buses available for baseload candidates")
    if CARRIER in n.carriers.index:
        raise ValueError(f"Carrier {CARRIER!r} already exists")

    nyears = n.snapshot_weightings.objective.sum() / 8760
    costs = load_costs(cost_file, copy.deepcopy(cost_config), max_hours, nyears)
    nuclear = costs.loc["nuclear"]
    district_heat = district_heat or {}
    if district_heat.get("enabled", False):
        heat_penalty, heat_limit = heat_coupling_parameters(district_heat)
    availability = pd.read_csv(availability_file, index_col="country")["factor"]
    if availability.empty or not availability.between(0, 1, inclusive="right").all():
        raise ValueError("Nuclear availability factors must be in (0, 1]")
    bus_availability = (
        n.buses.loc[buses, "country"].map(availability).fillna(availability.median())
    )

    n.add("Carrier", CARRIER, co2_emissions=0)
    n.add(
        "Generator",
        buses + " " + CARRIER,
        bus=buses,
        carrier=CARRIER,
        p_nom_extendable=True,
        p_nom_max=capacity_gw * 1e3 if capacity_mode == "fixed" else math.inf,
        p_max_pu=bus_availability.to_numpy(),
        p_min_pu=bus_availability.to_numpy() if must_run else 0.0,
        marginal_cost=variable_opex_eur_per_mwh,
        capital_cost=annualized_capex_eur_per_mw_year,
        efficiency=nuclear["efficiency"],
        lifetime=nuclear["lifetime"],
    )
    logger.info(
        "Offered %s baseload across %d AC buses; annualized CAPEX %.2f EUR/MW-year, "
        "variable OPEX %.2f EUR/MWh; nuclear-like availability, efficiency and lifetime",
        f"{capacity_gw:g} GW fixed" if capacity_mode == "fixed" else "endogenous",
        len(buses),
        annualized_capex_eur_per_mw_year,
        variable_opex_eur_per_mwh,
    )
    if must_run:
        logger.info("Baseload must run at the country availability factor in every snapshot")

    if district_heat.get("enabled", False):
        district_buses = n.buses.index[n.buses.carrier == "urban central heat"]
        heat_nodes = buses[(buses + " urban central heat").isin(district_buses)]
        if heat_nodes.empty:
            raise ValueError("No AC buses have a matching urban central heat bus")
        if HEAT_CARRIER in n.carriers.index:
            raise ValueError(f"Carrier {HEAT_CARRIER!r} already exists")

        n.add("Carrier", HEAT_CARRIER)
        n.add(
            "Link",
            heat_nodes + " " + HEAT_CARRIER,
            bus0=heat_nodes,
            bus1=heat_nodes + " urban central heat",
            carrier=HEAT_CARRIER,
            p_nom_extendable=True,
            p_nom_max=(
                capacity_gw * 1e3 * heat_penalty * heat_limit
                if capacity_mode == "fixed" else math.inf
            ),
            efficiency=1 / heat_penalty,
            marginal_cost=0.0,
            capital_cost=0.0,
            lifetime=nuclear["lifetime"],
        )
        logger.info(
            "Offered district heat at %d nodes: %.3f MWh electricity lost/MWh heat; "
            "heat at most %.1f%% of gross electric output",
            len(heat_nodes),
            heat_penalty,
            100 * heat_limit,
        )


def heat_coupling_parameters(options):
    """Return and validate the electricity penalty and heat/gross-power limit."""
    penalty = float(options["electricity_loss_per_mwh_heat"])
    limit = float(options["max_heat_per_mw_e"])
    if not 0 < penalty < 1:
        raise ValueError("Electricity loss per MWh of heat must be between 0 and 1")
    if limit <= 0 or penalty * limit > 1:
        raise ValueError("Heat limit must be positive and leave non-negative net electricity")
    return penalty, limit


def capex_opex_point(baseline_metrics, free_metrics, reference_gw, capex_share):
    """Price the free reference case on its average annual break-even line."""
    baseline = pd.read_csv(baseline_metrics).iloc[0]
    free = pd.read_csv(free_metrics).iloc[0]
    reference_gw = float(reference_gw)
    capex_share = float(capex_share)
    if not 0 < capex_share < 1 or reference_gw <= 0:
        raise ValueError("CAPEX share must be in (0, 1) and reference GW positive")
    if (
        baseline["scenario"] != free["scenario"]
        or baseline["weather_year"] != free["weather_year"]
        or not math.isclose(float(baseline["built_gw"]), 0, abs_tol=1e-3)
        or not math.isclose(float(free["built_gw"]), reference_gw, abs_tol=1e-3)
    ):
        raise ValueError("Baseline and free-case metrics do not match the reference")
    saving = float(baseline["system_cost_eur_per_year"] - free["system_cost_eur_per_year"])
    generation = float(free["baseload_generation_mwh"])
    if not all(math.isfinite(v) and v > 0 for v in (saving, generation)):
        raise ValueError("Reference saving and gross generation must be positive")
    annualized_capex = capex_share * saving / (reference_gw * 1e3)
    variable_opex = (1 - capex_share) * saving / generation
    return annualized_capex, variable_opex


def add_baseload_capacity_constraint(n, capacity_gw):
    """Fix the sum of candidate nameplate capacities, leaving siting optimal."""
    capacity_gw = float(capacity_gw)
    if capacity_gw == 0:
        return
    names = n.generators.index[n.generators.carrier == CARRIER]
    if names.empty:
        raise ValueError("Baseload capacity requested without candidate generators")
    capacity = n.model["Generator-p_nom"].loc[names].sum()
    n.model.add_constraints(
        capacity == capacity_gw * 1e3, name="baseload_candidate_total_capacity"
    )


def add_baseload_heat_constraints(n, options):
    """Tie each free heat link to its colocated baseload generator.

    A link consumes the foregone electricity from the AC bus and injects heat
    into the corresponding district-heating bus. Without these constraints it
    could act as an unrelated, free high-COP heat pump.
    """
    penalty, limit = heat_coupling_parameters(options)
    heat_links = n.links.index[n.links.carrier == HEAT_CARRIER]
    if heat_links.empty:
        raise ValueError("District-heat coupling enabled without baseload heat links")

    gross_power = n.model["Generator-p"]
    heat_electricity = n.model["Link-p"]
    gross_capacity = n.model["Generator-p_nom"]
    heat_capacity = n.model["Link-p_nom"]
    for number, link in enumerate(heat_links):
        generator = n.links.at[link, "bus0"] + " " + CARRIER
        if generator not in n.generators.index:
            raise ValueError(f"Missing baseload generator for {link}")
        ratio = penalty * limit
        n.model.add_constraints(
            heat_electricity.loc[:, link] <= ratio * gross_power.loc[:, generator],
            name=f"baseload_heat_dispatch_{number}",
        )
        n.model.add_constraints(
            heat_capacity.loc[link] == ratio * gross_capacity.loc[generator],
            name=f"baseload_heat_capacity_{number}",
        )


def write_baseload_metrics(
    n, output_file, requested_gw, run_name, capacity_mode="fixed",
    capex_share=None, annualized_capex_eur_per_mw_year=0.0,
    variable_opex_eur_per_mwh=0.0, must_run=False,
):
    """Write compact results so the sweep can be summarised without reloading networks."""
    names = n.generators.index[n.generators.carrier == CARRIER]
    built_mw = n.generators.loc[names, "p_nom_opt"].sum() if len(names) else 0.0
    requested_mw = float(requested_gw) * 1e3
    if capacity_mode == "fixed" and abs(built_mw - requested_mw) > max(
        1e-3, requested_mw * 1e-6
    ):
        raise ValueError(f"Solved baseload capacity {built_mw} MW differs from {requested_mw} MW")

    generation_mwh = 0.0
    fixed_om_eur = 0.0
    heat_mwh = 0.0
    electricity_loss_mwh = 0.0
    if len(names):
        generation_mwh = (
            n.generators_t.p[names]
            .mul(n.snapshot_weightings.generators, axis=0)
            .sum()
            .sum()
        )
        fixed_om_eur = (
            n.generators.loc[names, "p_nom_opt"]
            * n.generators.loc[names, "capital_cost"]
        ).sum()

    heat_links = n.links.index[n.links.carrier == HEAT_CARRIER]
    if len(heat_links):
        heat_mwh = -(
            n.links_t.p1[heat_links]
            .mul(n.snapshot_weightings.generators, axis=0)
            .sum()
            .sum()
        )
        electricity_loss_mwh = (
            n.links_t.p0[heat_links]
            .mul(n.snapshot_weightings.generators, axis=0)
            .sum()
            .sum()
        )

    row = {
        "scenario": run_name,
        "weather_year": pd.Timestamp(n.snapshots[0]).year,
        "baseload_gw": float(requested_gw),
        "built_gw": built_mw / 1e3,
        "system_cost_eur_per_year": n.objective,
        "baseload_generation_mwh": generation_mwh,
        "baseload_net_electricity_mwh": generation_mwh - electricity_loss_mwh,
        "baseload_district_heat_mwh": heat_mwh,
        "baseload_electricity_loss_mwh": electricity_loss_mwh,
        "baseload_fixed_om_eur_per_year": fixed_om_eur,
    }
    if capacity_mode == "endogenous":
        row.update(
            capacity_mode=capacity_mode,
            must_run=bool(must_run),
            reference_capacity_gw=float(requested_gw),
            baseload_gw=built_mw / 1e3,
            capex_share=float(capex_share),
            annualized_capex_eur_per_mw_year=annualized_capex_eur_per_mw_year,
            variable_opex_eur_per_mwh=variable_opex_eur_per_mwh,
            baseload_fixed_om_eur_per_year=0.0,
            baseload_annualized_capex_eur_per_year=fixed_om_eur,
            baseload_variable_opex_eur_per_year=(
                generation_mwh * variable_opex_eur_per_mwh
            ),
        )
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
