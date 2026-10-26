# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Capacity and accounting helpers for the baseload weather experiment."""

import copy
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
CARRIER = "baseload_candidate"


def add_baseload_generators(
    n, capacity_gw, cost_file, cost_config, max_hours, availability_file
):
    """Offer zero-cost capacity with nuclear-like availability at every AC bus.

    Existing nuclear plants remain unchanged. The experiment's exact total
    capacity is imposed in ``add_baseload_capacity_constraint``.
    """
    capacity_gw = float(capacity_gw)
    if capacity_gw < 0:
        raise ValueError("Baseload capacity must be non-negative")
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
        p_nom_max=capacity_gw * 1e3,
        p_max_pu=bus_availability.to_numpy(),
        marginal_cost=0.0,
        capital_cost=0.0,
        efficiency=nuclear["efficiency"],
        lifetime=nuclear["lifetime"],
    )
    logger.info(
        "Offered %.0f GW of zero-CAPEX, zero-OPEX baseload across %d AC buses; "
        "nuclear-like availability, efficiency and lifetime",
        capacity_gw,
        len(buses),
    )


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


def write_baseload_metrics(n, output_file, requested_gw, run_name):
    """Write compact results so the sweep can be summarised without reloading networks."""
    names = n.generators.index[n.generators.carrier == CARRIER]
    built_mw = n.generators.loc[names, "p_nom_opt"].sum() if len(names) else 0.0
    requested_mw = float(requested_gw) * 1e3
    if abs(built_mw - requested_mw) > max(1e-3, requested_mw * 1e-6):
        raise ValueError(f"Solved baseload capacity {built_mw} MW differs from {requested_mw} MW")

    generation_mwh = 0.0
    fixed_om_eur = 0.0
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

    row = {
        "scenario": run_name,
        "weather_year": pd.Timestamp(n.snapshots[0]).year,
        "baseload_gw": float(requested_gw),
        "built_gw": built_mw / 1e3,
        "system_cost_eur_per_year": n.objective,
        "baseload_generation_mwh": generation_mwh,
        "baseload_fixed_om_eur_per_year": fixed_om_eur,
    }
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
