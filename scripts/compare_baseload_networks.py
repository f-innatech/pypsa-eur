# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Compare solved baseload networks by cost, generation, siting, and district heat.

Example:
    python scripts/compare_baseload_networks.py results/weather-2010/baseload/networks \
        --output-dir results/weather-2010/baseload/comparison

Directories are searched recursively, so passing ``results`` compares all
downloaded weather years. Missing capacity cases are allowed. Each network is
loaded separately to limit memory use.
"""

import argparse
import gc
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pypsa


CAPACITY_IN_FILENAME = re.compile(r"_b(?P<gw>\d+(?:\.\d+)?)\.nc$")
# Link p_nom is the power at bus0, not output power at its other ports.
CAPACITY_COMPONENTS = (
    ("Generator", "generators", "p_nom", "p_nom_opt", "GW", "output power"),
    ("Link", "links", "p_nom", "p_nom_opt", "GW", "bus0 input power"),
    (
        "StorageUnit",
        "storage_units",
        "p_nom",
        "p_nom_opt",
        "GW",
        "discharge power",
    ),
    ("Store", "stores", "e_nom", "e_nom_opt", "GWh", "stored energy"),
)
CAPACITY_KEYS = ["component", "carrier", "unit", "capacity_basis"]
DISTRICT_HEAT_CARRIER = "urban central heat"
# These links move electricity or discharge storage; they are not generators.
NON_GENERATION_LINKS = {"DC", "electricity distribution grid", "battery discharger"}
TRANSPORT_FUEL_LOAD_CARRIERS = {
    "land transport oil", "kerosene for aviation", "shipping oil", "shipping methanol",
}
INDUSTRY_ENERGY_LOAD_CARRIERS = {
    "industry electricity", "H2 for industry", "gas for industry",
    "coal for industry", "solid biomass for industry",
    "low-temperature heat for industry",
}
OBJECTIVE_COMPONENTS = (
    ("Generator", "generators", "p_nom", "p"),
    ("Link", "links", "p_nom", "p0"),
    ("Store", "stores", "e_nom", "p"),
    ("StorageUnit", "storage_units", "p_nom", "p_dispatch"),
    ("Line", "lines", "s_nom", None),
    ("Transformer", "transformers", "s_nom", None),
)


def find_networks(inputs):
    files = []
    for item in inputs:
        if item.is_file():
            files.append(item)
        elif item.is_dir():
            files.extend(item.rglob("*.nc"))
        else:
            raise FileNotFoundError(item)
    files = sorted({path.resolve() for path in files if CAPACITY_IN_FILENAME.search(path.name)})
    if not files:
        raise ValueError("No solved baseload networks (*_b0.nc, *_b60.nc, ...) found")
    return files


def nominal_capacity(table, nominal, optimized):
    """Return solved capacity, falling back to fixed capacity where necessary."""
    if optimized in table:
        capacity = pd.to_numeric(table[optimized], errors="coerce")
        capacity = capacity.fillna(pd.to_numeric(table[nominal], errors="coerce"))
    else:
        capacity = pd.to_numeric(table[nominal], errors="coerce")
    if capacity.isna().any():
        raise ValueError(f"Missing optimized {optimized} capacity")
    return capacity


def active_assets(table):
    if "active" in table:
        return table.loc[table.active.fillna(True).astype(bool)]
    return table


def snapshot_weights(network, column="generators"):
    return network.snapshot_weightings[column].reindex(network.snapshots)


def link_efficiency(network, names, port):
    """Mean port efficiency; time-varying COPs use the weighted annual mean."""
    attribute = "efficiency" if port == 1 else f"efficiency{port}"
    static = pd.to_numeric(network.links.loc[names, attribute], errors="coerce")
    dynamic = getattr(network.links_t, attribute, pd.DataFrame())
    available = names.intersection(dynamic.columns)
    if len(available):
        weights = snapshot_weights(network)
        average = dynamic[available].mul(weights, axis=0).sum() / weights.sum()
        static.loc[available] = average.fillna(static.loc[available])
    return static.fillna(0)


def electric_generation_capacities(network, weather_year, baseload_gw):
    """Electricity-producing generators and links, measured at the AC output."""
    ac_buses = set(network.buses.index[network.buses.carrier == "AC"])
    rows = []
    generators = active_assets(network.generators)
    generators = generators.loc[generators.bus.isin(ac_buses)]
    generator_power = nominal_capacity(generators, "p_nom", "p_nom_opt") / 1e3
    for carrier, capacity in generator_power.groupby(generators.carrier).sum().items():
        rows.append(
            dict(weather_year=weather_year, baseload_gw=baseload_gw,
                 component="Generator", carrier=carrier,
                 installed_capacity_gw_e=capacity)
        )

    links = active_assets(network.links)
    links = links.loc[~links.carrier.isin(NON_GENERATION_LINKS)]
    link_power = nominal_capacity(links, "p_nom", "p_nom_opt")
    for port in range(1, 5):
        bus = f"bus{port}"
        if bus not in links:
            continue
        names = links.index[links[bus].isin(ac_buses)]
        if len(names) == 0:
            continue
        efficiency = link_efficiency(network, names, port)
        output = link_power.loc[names] * efficiency.clip(lower=0) / 1e3
        for carrier, capacity in output.groupby(links.loc[names, "carrier"]).sum().items():
            if capacity > 1e-9:
                rows.append(
                    dict(weather_year=weather_year, baseload_gw=baseload_gw,
                         component="Link", carrier=carrier,
                         installed_capacity_gw_e=capacity)
                )
    return rows


def electricity_generation_energy(network, weather_year, baseload_gw):
    """Annual electric output, excluding transmission and storage discharge."""
    electric_buses = set(
        network.buses.index[network.buses.carrier.isin(["AC", "low voltage"])]
    )
    weights = snapshot_weights(network)
    rows = []

    generators = active_assets(network.generators)
    generators = generators.loc[generators.bus.isin(electric_buses)]
    generation = (
        network.generators_t.p.reindex(
            index=network.snapshots, columns=generators.index, fill_value=0
        )
        .clip(lower=0)
        .mul(weights, axis=0)
        .sum() / 1e6
    )
    for carrier, energy in generation.groupby(generators.carrier).sum().items():
        rows.append(dict(weather_year=weather_year, baseload_gw=baseload_gw,
                         component="Generator", carrier=carrier,
                         generation_twh=energy))

    # Reservoir hydro uses natural inflow; pumped hydro is storage discharge.
    hydro = active_assets(network.storage_units)
    hydro = hydro.loc[(hydro.carrier == "hydro") & hydro.bus.isin(electric_buses)]
    if not hydro.empty:
        hydro_generation = (
            network.storage_units_t.p_dispatch.reindex(
                index=network.snapshots, columns=hydro.index, fill_value=0
            )
            .clip(lower=0)
            .mul(weights, axis=0)
            .sum().sum() / 1e6
        )
        rows.append(dict(weather_year=weather_year, baseload_gw=baseload_gw,
                         component="StorageUnit", carrier="hydro",
                         generation_twh=hydro_generation))

    excluded = NON_GENERATION_LINKS | {"home battery discharger", "V2G"}
    links = active_assets(network.links)
    links = links.loc[~links.carrier.isin(excluded)]
    for port in range(1, 5):
        bus = f"bus{port}"
        if bus not in links:
            continue
        names = links.index[links[bus].isin(electric_buses)]
        if len(names) == 0:
            continue
        flow = getattr(network.links_t, f"p{port}", pd.DataFrame())
        if not names.isin(flow.columns).all():
            raise ValueError(f"Missing electricity-output dispatch for Link port {port}")
        output = (
            (-flow[names]).clip(lower=0)
            .mul(weights, axis=0)
            .sum() / 1e6
        )
        for carrier, energy in output.groupby(links.loc[names, "carrier"]).sum().items():
            rows.append(dict(weather_year=weather_year, baseload_gw=baseload_gw,
                             component="Link", carrier=carrier,
                             generation_twh=energy))
    return rows


def sector_final_energy_demand(network, weather_year, baseload_gw):
    """Final energy at end-use loads and decentral heating-supply inputs."""
    weights = snapshot_weights(network)
    rows = []

    def annual_twh(dispatch, names):
        if len(names) == 0:
            return pd.Series(dtype=float)
        if not names.isin(dispatch.columns).all():
            raise ValueError("Missing sector demand dispatch time series")
        return dispatch[names].mul(weights, axis=0).sum() / 1e6

    def add_rows(sector, values, carriers, basis):
        for carrier, value in values.groupby(carriers).sum().items():
            rows.append(dict(weather_year=weather_year, baseload_gw=baseload_gw,
                             sector=sector, carrier=carrier, basis=basis,
                             demand_twh=value))

    loads = network.loads
    transport_fuels = loads.index[loads.carrier.isin(TRANSPORT_FUEL_LOAD_CARRIERS)]
    add_rows("Transport", annual_twh(network.loads_t.p, transport_fuels),
             loads.loc[transport_fuels, "carrier"], "fuel at end-use load")
    industry = loads.index[loads.carrier.isin(INDUSTRY_ENERGY_LOAD_CARRIERS)]
    add_rows("Industry", annual_twh(network.loads_t.p, industry),
             loads.loc[industry, "carrier"], "energy at end-use load")

    links = active_assets(network.links)
    bev = links.index[links.carrier == "BEV charger"]
    v2g = links.index[links.carrier == "V2G"]
    # Link p1 is negative when V2G injects electricity at the low-voltage bus.
    ev_net = annual_twh(network.links_t.p0, bev).sum()
    ev_net += annual_twh(network.links_t.p1, v2g).sum()
    rows.append(dict(weather_year=weather_year, baseload_gw=baseload_gw,
                     sector="Transport", carrier="BEV net grid electricity",
                     basis="charger input minus V2G output", demand_twh=ev_net))

    district = loads.index[loads.carrier == "urban central heat"]
    district_loss = network.meta.get("sector", {}).get("district_heating", {}).get(
        "district_heating_loss", 0
    )
    district_heat = annual_twh(network.loads_t.p, district).sum() / (1 + district_loss)
    rows.append(dict(weather_year=weather_year, baseload_gw=baseload_gw,
                     sector="Heating", carrier="district heat delivered",
                     basis="heat load without distribution loss markup",
                     demand_twh=district_heat))

    decentral_buses = set(network.buses.index[
        network.buses.carrier.isin(["rural heat", "urban decentral heat"])
    ])
    heating_links = links.carrier.str.contains(
        "heat pump|gas boiler|biomass boiler|resistive heater", regex=True
    )
    heat_output = pd.Series(False, index=links.index)
    for port in range(1, 5):
        bus = f"bus{port}"
        if bus in links:
            heat_output |= links[bus].isin(decentral_buses)
    names = links.index[heating_links & heat_output]
    add_rows("Heating", annual_twh(network.links_t.p0, names),
             links.loc[names, "carrier"], "electricity or fuel input")

    solar = active_assets(network.generators)
    solar = solar.loc[
        solar.bus.isin(decentral_buses) & solar.carrier.str.contains("solar thermal")
    ]
    add_rows("Heating", annual_twh(network.generators_t.p, solar.index),
             solar.carrier, "solar heat output")
    return rows


def sector_load_profile(network):
    """Europe-wide sector load at each snapshot, using the annual chart's basis."""
    snapshots = network.snapshots

    def total(dispatch, names):
        if len(names) == 0:
            return pd.Series(0.0, index=snapshots)
        if not names.isin(dispatch.columns).all():
            raise ValueError("Missing sector load time series")
        return dispatch[names].sum(axis=1).reindex(snapshots)

    loads = network.loads
    links = active_assets(network.links)
    transport_fuels = loads.index[loads.carrier.isin(TRANSPORT_FUEL_LOAD_CARRIERS)]
    industry = loads.index[loads.carrier.isin(INDUSTRY_ENERGY_LOAD_CARRIERS)]
    other_electricity = loads.index[loads.carrier == "electricity"]
    bev = links.index[links.carrier == "BEV charger"]
    v2g = links.index[links.carrier == "V2G"]

    district = loads.index[loads.carrier == "urban central heat"]
    district_loss = network.meta.get("sector", {}).get("district_heating", {}).get(
        "district_heating_loss", 0
    )
    decentral_buses = set(network.buses.index[
        network.buses.carrier.isin(["rural heat", "urban decentral heat"])
    ])
    heating_links = links.carrier.str.contains(
        "heat pump|gas boiler|biomass boiler|resistive heater", regex=True
    )
    heat_output = pd.Series(False, index=links.index)
    for port in range(1, 5):
        bus = f"bus{port}"
        if bus in links:
            heat_output |= links[bus].isin(decentral_buses)
    heating_names = links.index[heating_links & heat_output]
    solar = active_assets(network.generators)
    solar = solar.loc[
        solar.bus.isin(decentral_buses) & solar.carrier.str.contains("solar thermal")
    ]

    profile = pd.DataFrame(index=snapshots)
    profile["Transport"] = (
        total(network.loads_t.p, transport_fuels)
        + total(network.links_t.p0, bev)
        + total(network.links_t.p1, v2g)
    )
    profile["Industry"] = total(network.loads_t.p, industry)
    profile["Heating"] = (
        total(network.loads_t.p, district) / (1 + district_loss)
        + total(network.links_t.p0, heating_names)
        + total(network.generators_t.p, solar.index)
    )
    profile["Other electricity"] = total(network.loads_t.p, other_electricity)
    profile.index.name = "Time"
    return profile / 1e3  # MW to GW


def baseload_siting(network, weather_year, baseload_gw):
    """Installed experimental baseload and annual output at each AC bus."""
    buses = network.buses.loc[network.buses.carrier == "AC"]
    candidates = active_assets(network.generators)
    candidates = candidates.loc[candidates.carrier == "baseload_candidate"]
    if not candidates.bus.isin(buses.index).all():
        raise ValueError("Baseload candidate attached to a non-AC bus")

    capacity_mw = nominal_capacity(candidates, "p_nom", "p_nom_opt")
    if abs(capacity_mw.sum() / 1e3 - baseload_gw) > max(1e-6, baseload_gw * 1e-6):
        raise ValueError(f"Solved baseload capacity differs from {baseload_gw:g} GW")
    dispatch = network.generators_t.p
    if not candidates.index.isin(dispatch.columns).all():
        raise ValueError("Missing baseload dispatch time series")
    generation_mwh = (
        dispatch[candidates.index]
        .mul(snapshot_weights(network), axis=0)
        .sum(axis=0)
    )

    siting = pd.DataFrame(index=buses.index)
    siting.index.name = "node"
    siting["country"] = buses.country.replace("", pd.NA).fillna("unknown")
    siting["bus_x"] = buses.x
    siting["bus_y"] = buses.y
    siting["installed_capacity_gw"] = (
        (capacity_mw.groupby(candidates.bus).sum() / 1e3)
        .reindex(siting.index, fill_value=0)
    )
    siting["generation_twh"] = (
        (generation_mwh.groupby(candidates.bus).sum() / 1e6)
        .reindex(siting.index, fill_value=0)
    )
    siting["capacity_share_percent"] = (
        siting.installed_capacity_gw / baseload_gw * 100 if baseload_gw else 0.0
    )
    siting["full_load_hours"] = (
        siting.generation_twh * 1e3
        / siting.installed_capacity_gw.where(siting.installed_capacity_gw > 1e-9)
    )
    siting.insert(0, "baseload_gw", baseload_gw)
    siting.insert(0, "weather_year", weather_year)
    return siting.reset_index()


def installed_capacities(network, weather_year, baseload_gw):
    rows = []
    for component, attribute, nominal, optimized, unit, basis in CAPACITY_COMPONENTS:
        table = getattr(network, attribute)
        if table.empty:
            continue
        table = active_assets(table)
        capacity = nominal_capacity(table, nominal, optimized)

        if "carrier" in table:
            carrier = table.carrier.replace("", pd.NA).fillna("unspecified")
        else:
            carrier = pd.Series("unspecified", index=table.index)
        totals = (
            pd.DataFrame({"carrier": carrier, "capacity": capacity / 1e3})
            .groupby("carrier", sort=True)["capacity"]
            .sum()
        )
        for name, value in totals.items():
            if abs(value) < 1e-9:
                continue
            rows.append(
                {
                    "weather_year": weather_year,
                    "baseload_gw": baseload_gw,
                    "component": component,
                    "carrier": name,
                    "unit": unit,
                    "capacity_basis": basis,
                    "installed_capacity": value,
                }
            )
    return rows


def annual_variable_cost(network, component, names, dispatch_attribute,
                         cost_attribute="marginal_cost"):
    """Weighted cost of one dispatch variable; upstream costs are separate."""
    if len(names) == 0:
        return pd.Series(dtype=float)
    table = getattr(network, component)
    pnl = getattr(network, f"{component}_t")
    if cost_attribute not in table:
        return pd.Series(0.0, index=names)
    dispatch = getattr(pnl, dispatch_attribute).reindex(
        index=network.snapshots, columns=names, fill_value=0
    )
    weighted = dispatch.mul(snapshot_weights(network, "objective"), axis=0)
    static = pd.to_numeric(table.loc[names, cost_attribute], errors="coerce").fillna(0)
    result = weighted.sum() * static
    dynamic = getattr(pnl, cost_attribute, pd.DataFrame())
    varying = names.intersection(dynamic.columns)
    if len(varying):
        rates = dynamic[varying].reindex(network.snapshots)
        rates = rates.fillna(static.loc[varying])
        result.loc[varying] = (weighted[varying] * rates).sum()
    return result


def system_cost_category(network, component, table):
    """Assign each asset's full objective contribution to one functional group."""
    if component in {"Line", "Transformer"}:
        return pd.Series("Infrastructure", index=table.index)
    if component == "StorageUnit":
        return pd.Series("Storage", index=table.index)
    carrier = table.carrier.fillna("")
    if component == "Store":
        return pd.Series(
            np.where(carrier.str.contains("co2", case=False), "Carbon management", "Storage"),
            index=table.index,
        )
    if component == "Generator":
        bus_carrier = table.bus.map(network.buses.carrier).fillna("")
        return pd.Series(
            np.select(
                [bus_carrier.isin(["AC", "low voltage"]),
                 bus_carrier.str.contains("heat", case=False)],
                ["Electricity generation", "Heat supply"],
                default="Fuel and resources",
            ),
            index=table.index,
        )

    # Multi-output links are counted once. CHP is assigned to electricity
    # generation; its heat output does not create a second cost entry.
    group = pd.Series("Conversion and industry", index=table.index)
    electric_output = pd.Series(False, index=table.index)
    heat_output = pd.Series(False, index=table.index)
    for port in range(1, 5):
        bus = f"bus{port}"
        if bus not in table:
            continue
        efficiency = "efficiency" if port == 1 else f"efficiency{port}"
        if efficiency not in table:
            continue
        producing = pd.to_numeric(table[efficiency], errors="coerce").fillna(0) > 0
        output_carrier = table[bus].map(network.buses.carrier).fillna("")
        electric_output |= producing & output_carrier.isin(["AC", "low voltage"])
        heat_output |= producing & output_carrier.str.contains("heat", case=False)
    group.loc[heat_output] = "Heat supply"
    group.loc[electric_output] = "Electricity generation"
    group.loc[carrier.str.contains(
        r"electrolysis|Fischer-Tropsch|methanolisation|methanation|SMR|for industry",
        case=False, regex=True,
    )] = "Conversion and industry"
    group.loc[carrier.str.contains(
        r"^DAC$|process emissions|co2|carbon capture|cement capture",
        case=False, regex=True,
    )] = "Carbon management"
    group.loc[carrier.str.contains(
        r"battery|water tanks|V2G|BEV charger|storage charger|storage discharger",
        case=False, regex=True,
    )] = "Storage"
    group.loc[carrier.str.contains(
        r"^DC$|pipeline|distribution grid|grid connection|transmission",
        case=False, regex=True,
    )] = "Infrastructure"
    return group


def system_cost_breakdown(network, weather_year, baseload_gw):
    """Reconstruct PyPSA's objective by component and carrier, in EUR/year."""
    rows = []
    for component, attribute, nominal, dispatch_attribute in OBJECTIVE_COMPONENTS:
        table = active_assets(getattr(network, attribute))
        if table.empty:
            continue
        fixed = pd.Series(0.0, index=table.index)
        extendable = table[f"{nominal}_extendable"].fillna(False).astype(bool)
        if extendable.any():
            capacity = nominal_capacity(table, nominal, f"{nominal}_opt")
            fixed.loc[extendable] = (
                (capacity.loc[extendable] - table.loc[extendable, nominal])
                * table.loc[extendable, "capital_cost"]
            )

        variable = pd.Series(0.0, index=table.index)
        if dispatch_attribute is not None:
            # Keep peak memory bounded when a network has thousands of links.
            for start in range(0, len(table), 256):
                names = table.index[start:start + 256]
                variable.loc[names] += annual_variable_cost(
                    network, attribute, names, dispatch_attribute
                )
                if component in {"Store", "StorageUnit"}:
                    extra = (
                        (("marginal_cost_storage", "e"),) if component == "Store"
                        else (("marginal_cost_storage", "state_of_charge"),
                              ("spill_cost", "spill"))
                    )
                    for cost_attribute, operation in extra:
                        if (cost_attribute not in table
                                or not table.loc[names, cost_attribute].fillna(0).ne(0).any()):
                            continue
                        variable.loc[names] += annual_variable_cost(
                            network, attribute, names, operation, cost_attribute
                        )
                if component in {"Generator", "Link"}:
                    if ("stand_by_cost" in table
                            and table.loc[names, "stand_by_cost"].fillna(0).ne(0).any()):
                        variable.loc[names] += annual_variable_cost(
                            network, attribute, names, "status", "stand_by_cost"
                        )
                    for operation in ("start_up", "shut_down"):
                        cost_attribute = f"{operation}_cost"
                        if (cost_attribute not in table
                                or not table.loc[names, cost_attribute].fillna(0).ne(0).any()):
                            continue
                        events = getattr(getattr(network, f"{attribute}_t"), operation)
                        variable.loc[names] += events.reindex(
                            index=network.snapshots, columns=names, fill_value=0
                        ).sum().mul(table.loc[names, cost_attribute])

        if ("marginal_cost_quadratic" in table
                and table.marginal_cost_quadratic.fillna(0).ne(0).any()):
            raise ValueError("Quadratic marginal costs need separate objective accounting")
        costs = pd.DataFrame({
            "category": system_cost_category(network, component, table),
            "carrier": table.carrier.replace("", pd.NA).fillna("unspecified"),
            "investment_cost_eur_per_year": fixed,
            "operating_cost_eur_per_year": variable,
        })
        costs = costs.loc[
            (costs.investment_cost_eur_per_year.abs() > 1e-4)
            | (costs.operating_cost_eur_per_year.abs() > 1e-4)
        ]
        if costs.empty:
            continue
        costs["component"] = component
        rows.append(costs.groupby(["category", "component", "carrier"], as_index=False)
                    [["investment_cost_eur_per_year", "operating_cost_eur_per_year"]].sum())

    breakdown = pd.concat(rows, ignore_index=True)
    breakdown["total_cost_eur_per_year"] = (
        breakdown.investment_cost_eur_per_year + breakdown.operating_cost_eur_per_year
    )
    objective = float(network.objective)
    residual = objective - breakdown.total_cost_eur_per_year.sum()
    if abs(residual) > max(1.0, abs(objective) * 1e-9):
        print(f"Warning: {weather_year} b{baseload_gw:g} objective residual "
              f"{residual / 1e9:+.4f} billion EUR/year")
        breakdown.loc[len(breakdown)] = dict(
            category="Other objective terms", component="Other", carrier="unallocated",
            investment_cost_eur_per_year=0.0,
            operating_cost_eur_per_year=residual,
            total_cost_eur_per_year=residual,
        )
    breakdown.insert(0, "baseload_gw", baseload_gw)
    breakdown.insert(0, "weather_year", weather_year)
    return breakdown


def district_heat_role(carrier):
    if "water tanks charger" in carrier:
        return "storage_charge", "district_heat_only"
    if "water tanks discharger" in carrier:
        return "storage_discharge", "district_heat_only"
    if "CHP" in carrier or carrier == "baseload_candidate_heat":
        return "chp_heat", "shared_chp"
    if "heat vent" in carrier:
        return "heat_vent", "district_heat_only"
    if carrier.startswith("urban central") or carrier == "geothermal district heat":
        return "heat_supply", "district_heat_only"
    return "process_heat", "shared_process"


def district_heating(network, weather_year, baseload_gw):
    """Extract every component connected to the district-heat bus system.

    PyPSA link p_i is positive when withdrawing from bus_i, negative when
    injecting into it. Shared CHP/process asset costs are reported in full
    but excluded from the dedicated-heat cost total.
    """
    heat_buses = set(network.buses.index[network.buses.carrier == DISTRICT_HEAT_CARRIER])
    if not heat_buses:
        return [], [], None
    weights = snapshot_weights(network)
    base = dict(weather_year=weather_year, baseload_gw=baseload_gw)
    asset_rows = []

    generators = active_assets(network.generators)
    generators = generators.loc[generators.bus.isin(heat_buses)]
    if not generators.empty:
        names = generators.index
        capacity = nominal_capacity(generators, "p_nom", "p_nom_opt")
        dispatch = network.generators_t.p.reindex(
            index=network.snapshots, columns=names, fill_value=0
        )
        injected = dispatch.clip(lower=0).mul(weights, axis=0).sum() / 1e6
        withdrawn = (-dispatch).clip(lower=0).mul(weights, axis=0).sum() / 1e6
        variable = annual_variable_cost(network, "generators", names, "p")
        for name in names:
            role, scope = district_heat_role(generators.at[name, "carrier"])
            asset_rows.append(dict(
                base, component="Generator", carrier=generators.at[name, "carrier"],
                role=role, cost_scope=scope, installed_input_capacity_gw=0.0,
                installed_heat_output_capacity_gw_th=capacity[name] / 1e3 if role != "heat_vent" else 0.0,
                installed_heat_withdrawal_capacity_gw_th=capacity[name] / 1e3 if role == "heat_vent" else 0.0,
                storage_energy_capacity_gwh_th=0.0,
                heat_injected_twh_th=injected[name], heat_withdrawn_twh_th=withdrawn[name],
                annualized_fixed_cost_bn_eur=capacity[name] * generators.at[name, "capital_cost"] / 1e9,
                direct_variable_cost_bn_eur=variable[name] / 1e9,
            ))

    links = active_assets(network.links)
    ports = [i for i in range(5) if f"bus{i}" in links]
    connected = pd.Series(False, index=links.index)
    for port in ports:
        connected |= links[f"bus{port}"].isin(heat_buses)
    links = links.loc[connected]
    if not links.empty:
        names = links.index
        capacity = nominal_capacity(links, "p_nom", "p_nom_opt")
        injected = pd.Series(0.0, index=names)
        withdrawn = pd.Series(0.0, index=names)
        output_capacity = pd.Series(0.0, index=names)
        withdrawal_capacity = pd.Series(0.0, index=names)
        for port in ports:
            attached = names[links[f"bus{port}"].isin(heat_buses)]
            if len(attached) == 0:
                continue
            flow = getattr(network.links_t, f"p{port}").reindex(
                index=network.snapshots, columns=attached, fill_value=0
            )
            injected.loc[attached] += (-flow).clip(lower=0).mul(weights, axis=0).sum() / 1e6
            withdrawn.loc[attached] += flow.clip(lower=0).mul(weights, axis=0).sum() / 1e6
            if port == 0:
                withdrawal_capacity.loc[attached] += capacity.loc[attached] / 1e3
            else:
                efficiency = link_efficiency(network, attached, port)
                output_capacity.loc[attached] += (
                    capacity.loc[attached] * efficiency.clip(lower=0) / 1e3
                )
                withdrawal_capacity.loc[attached] += (
                    capacity.loc[attached] * (-efficiency).clip(lower=0) / 1e3
                )
        variable = annual_variable_cost(network, "links", names, "p0")
        for name in names:
            role, scope = district_heat_role(links.at[name, "carrier"])
            asset_rows.append(dict(
                base, component="Link", carrier=links.at[name, "carrier"],
                role=role, cost_scope=scope,
                installed_input_capacity_gw=capacity[name] / 1e3,
                installed_heat_output_capacity_gw_th=output_capacity[name],
                installed_heat_withdrawal_capacity_gw_th=withdrawal_capacity[name],
                storage_energy_capacity_gwh_th=0.0,
                heat_injected_twh_th=injected[name], heat_withdrawn_twh_th=withdrawn[name],
                annualized_fixed_cost_bn_eur=capacity[name] * links.at[name, "capital_cost"] / 1e9,
                direct_variable_cost_bn_eur=variable[name] / 1e9,
            ))

    stores = active_assets(network.stores)
    stores = stores.loc[
        stores.bus.isin(heat_buses)
        | stores.carrier.str.startswith("urban central water tanks", na=False)
    ]
    if not stores.empty:
        capacity = nominal_capacity(stores, "e_nom", "e_nom_opt")
        variable = annual_variable_cost(network, "stores", stores.index, "p")
        for name in stores.index:
            asset_rows.append(dict(
                base, component="Store", carrier=stores.at[name, "carrier"],
                role="thermal_storage_energy", cost_scope="district_heat_only",
                installed_input_capacity_gw=0.0, installed_heat_output_capacity_gw_th=0.0,
                installed_heat_withdrawal_capacity_gw_th=0.0,
                storage_energy_capacity_gwh_th=capacity[name] / 1e3,
                heat_injected_twh_th=0.0, heat_withdrawn_twh_th=0.0,
                annualized_fixed_cost_bn_eur=capacity[name] * stores.at[name, "capital_cost"] / 1e9,
                direct_variable_cost_bn_eur=variable[name] / 1e9,
            ))

    storage_units = active_assets(network.storage_units)
    storage_units = storage_units.loc[storage_units.bus.isin(heat_buses)]
    if not storage_units.empty:
        names = storage_units.index
        capacity = nominal_capacity(storage_units, "p_nom", "p_nom_opt")
        dispatch = network.storage_units_t.p.reindex(
            index=network.snapshots, columns=names, fill_value=0
        )
        injected = dispatch.clip(lower=0).mul(weights, axis=0).sum() / 1e6
        withdrawn = (-dispatch).clip(lower=0).mul(weights, axis=0).sum() / 1e6
        variable = annual_variable_cost(network, "storage_units", names, "p")
        for name in names:
            asset_rows.append(dict(
                base, component="StorageUnit", carrier=storage_units.at[name, "carrier"],
                role="thermal_storage_unit", cost_scope="district_heat_only",
                installed_input_capacity_gw=capacity[name] / 1e3,
                installed_heat_output_capacity_gw_th=capacity[name] / 1e3,
                installed_heat_withdrawal_capacity_gw_th=capacity[name] / 1e3,
                storage_energy_capacity_gwh_th=(
                    capacity[name] * storage_units.at[name, "max_hours"] / 1e3
                ),
                heat_injected_twh_th=injected[name], heat_withdrawn_twh_th=withdrawn[name],
                annualized_fixed_cost_bn_eur=(
                    capacity[name] * storage_units.at[name, "capital_cost"] / 1e9
                ),
                direct_variable_cost_bn_eur=variable[name] / 1e9,
            ))

    loads = active_assets(network.loads)
    loads = loads.loc[loads.bus.isin(heat_buses)]
    load_rows = []
    district_loss = (
        network.meta.get("sector", {}).get("district_heating", {}).get("district_heating_loss", 0)
    )
    if not loads.empty:
        demand = network.loads_t.p.reindex(
            index=network.snapshots, columns=loads.index, fill_value=0
        ).mul(weights, axis=0).sum() / 1e6
        for carrier, heat in demand.groupby(loads.carrier).sum().items():
            delivered = heat / (1 + district_loss) if carrier == DISTRICT_HEAT_CARRIER else heat
            load_rows.append(dict(
                base, carrier=carrier, heat_at_bus_twh_th=heat,
                estimated_delivered_heat_twh_th=delivered,
                modeled_distribution_loss_fraction=district_loss if carrier == DISTRICT_HEAT_CARRIER else 0,
            ))

    assets = pd.DataFrame(asset_rows)
    load_table = pd.DataFrame(load_rows)
    dedicated = assets.loc[assets.cost_scope == "district_heat_only"]
    summary = dict(
        base,
        heat_at_bus_twh_th=load_table.heat_at_bus_twh_th.sum(),
        estimated_delivered_heat_twh_th=load_table.estimated_delivered_heat_twh_th.sum(),
        primary_heat_injected_twh_th=assets.loc[
            assets.role.isin(["heat_supply", "chp_heat", "process_heat"]),
            "heat_injected_twh_th",
        ].sum(),
        dedicated_heat_injected_twh_th=assets.loc[
            assets.role == "heat_supply", "heat_injected_twh_th"
        ].sum(),
        chp_heat_injected_twh_th=assets.loc[
            assets.role == "chp_heat", "heat_injected_twh_th"
        ].sum(),
        shared_process_heat_injected_twh_th=assets.loc[
            assets.role == "process_heat", "heat_injected_twh_th"
        ].sum(),
        process_heat_consumption_twh_th=assets.loc[
            assets.role == "process_heat", "heat_withdrawn_twh_th"
        ].sum(),
        storage_charge_twh_th=assets.loc[
            assets.role.isin(["storage_charge", "thermal_storage_unit"]),
            "heat_withdrawn_twh_th",
        ].sum(),
        storage_discharge_twh_th=assets.loc[
            assets.role.isin(["storage_discharge", "thermal_storage_unit"]),
            "heat_injected_twh_th",
        ].sum(),
        heat_vent_twh_th=assets.loc[assets.role == "heat_vent", "heat_withdrawn_twh_th"].sum(),
        modeled_distribution_loss_twh_th=(
            load_table.heat_at_bus_twh_th.sum()
            - load_table.estimated_delivered_heat_twh_th.sum()
        ),
        heat_balance_residual_twh_th=(
            assets.heat_injected_twh_th.sum()
            - assets.heat_withdrawn_twh_th.sum()
            - load_table.heat_at_bus_twh_th.sum()
        ),
        dedicated_heat_asset_fixed_cost_bn_eur=dedicated.annualized_fixed_cost_bn_eur.sum(),
        dedicated_heat_asset_variable_cost_bn_eur=dedicated.direct_variable_cost_bn_eur.sum(),
        shared_chp_full_asset_cost_bn_eur=assets.loc[
            assets.cost_scope == "shared_chp",
            ["annualized_fixed_cost_bn_eur", "direct_variable_cost_bn_eur"],
        ].sum().sum(),
        shared_process_full_asset_cost_bn_eur=assets.loc[
            assets.cost_scope == "shared_process",
            ["annualized_fixed_cost_bn_eur", "direct_variable_cost_bn_eur"],
        ].sum().sum(),
        modeled_distribution_loss_fraction=district_loss,
        thermal_storage_capacity_gwh_th=assets.storage_energy_capacity_gwh_th.sum(),
    )
    return asset_rows, load_rows, summary


def compare(network_files):
    costs = []
    cost_details = []
    capacities = []
    electric_generation = []
    electricity_energy = []
    final_energy_demand = []
    sector_profiles = {}
    baseload_nodes = []
    heat_assets = []
    heat_loads = []
    heat_summaries = []
    seen = set()
    for path in network_files:
        baseload_gw = float(CAPACITY_IN_FILENAME.search(path.name).group("gw"))
        network = pypsa.Network(path)
        weather_year = pd.Timestamp(network.snapshots[0]).year
        key = (weather_year, baseload_gw)
        if key in seen:
            raise ValueError(f"Duplicate weather year and baseload capacity: {key}")
        seen.add(key)
        objective = float(network.objective)
        if not math.isfinite(objective):
            raise ValueError(f"Network has no finite objective: {path}")
        costs.append(
            {
                "weather_year": weather_year,
                "baseload_gw": baseload_gw,
                "system_cost_eur_per_year": objective,
                "network_file": str(path),
            }
        )
        cost_details.append(system_cost_breakdown(network, weather_year, baseload_gw))
        capacities.extend(installed_capacities(network, weather_year, baseload_gw))
        electric_generation.extend(
            electric_generation_capacities(network, weather_year, baseload_gw)
        )
        electricity_energy.extend(
            electricity_generation_energy(network, weather_year, baseload_gw)
        )
        final_energy_demand.extend(
            sector_final_energy_demand(network, weather_year, baseload_gw)
        )
        if weather_year not in sector_profiles or baseload_gw == 0:
            sector_profiles[weather_year] = (
                baseload_gw, sector_load_profile(network)
            )
        baseload_nodes.append(baseload_siting(network, weather_year, baseload_gw))
        assets, loads, summary = district_heating(network, weather_year, baseload_gw)
        heat_assets.extend(assets)
        heat_loads.extend(loads)
        if summary is not None:
            heat_summaries.append(summary)
        print(f"Loaded {path.name}: {objective / 1e9:.2f} billion EUR/year")
        del network
        gc.collect()

    costs = pd.DataFrame(costs).sort_values(["weather_year", "baseload_gw"])
    baseline_cost = costs.loc[costs.baseload_gw == 0].set_index("weather_year")
    costs["baseline_cost_eur_per_year"] = costs.weather_year.map(
        baseline_cost.system_cost_eur_per_year
    )
    costs["saving_vs_baseline_eur_per_year"] = (
        costs.baseline_cost_eur_per_year - costs.system_cost_eur_per_year
    )
    cost_detail_table = pd.concat(cost_details, ignore_index=True).sort_values(
        ["weather_year", "baseload_gw", "category", "component", "carrier"]
    )

    capacity_tables = []
    raw = pd.DataFrame(capacities)
    for weather_year, group in costs.groupby("weather_year"):
        cases = group.baseload_gw.tolist()
        year_capacities = raw.loc[raw.weather_year == weather_year]
        wide = year_capacities.pivot_table(
            index=CAPACITY_KEYS,
            columns="baseload_gw",
            values="installed_capacity",
            aggfunc="sum",
            fill_value=0,
        ).reindex(columns=cases, fill_value=0)
        baseline = wide[0.0] if 0.0 in wide else pd.Series(float("nan"), index=wide.index)
        for baseload_gw in cases:
            table = wide[[baseload_gw]].rename(
                columns={baseload_gw: "installed_capacity"}
            ).reset_index()
            table.insert(0, "weather_year", weather_year)
            table.insert(1, "baseload_gw", baseload_gw)
            table["baseline_capacity"] = baseline.to_numpy()
            table["delta_vs_baseline"] = (
                table.installed_capacity - table.baseline_capacity
            )
            capacity_tables.append(table)
    capacity_table = pd.concat(capacity_tables, ignore_index=True).sort_values(
        ["weather_year", "baseload_gw", "component", "carrier"]
    )
    generation_table = pd.DataFrame(electric_generation)
    generation_table = generation_table.groupby(
        ["weather_year", "baseload_gw", "component", "carrier"], as_index=False
    ).installed_capacity_gw_e.sum()
    baseline_generation = generation_table.loc[
        generation_table.baseload_gw == 0,
        ["weather_year", "component", "carrier", "installed_capacity_gw_e"],
    ].rename(columns={"installed_capacity_gw_e": "baseline_capacity_gw_e"})
    generation_table = generation_table.merge(
        baseline_generation, how="left", on=["weather_year", "component", "carrier"]
    )
    has_baseline = generation_table.weather_year.isin(baseline_generation.weather_year)
    generation_table.loc[has_baseline, "baseline_capacity_gw_e"] = (
        generation_table.loc[has_baseline, "baseline_capacity_gw_e"].fillna(0)
    )
    generation_table["delta_vs_baseline_gw_e"] = (
        generation_table.installed_capacity_gw_e - generation_table.baseline_capacity_gw_e
    )
    energy_table = pd.DataFrame(electricity_energy).groupby(
        ["weather_year", "baseload_gw", "component", "carrier"], as_index=False
    ).generation_twh.sum()
    energy_table = energy_table.sort_values(
        ["weather_year", "baseload_gw", "component", "carrier"]
    )
    demand_table = pd.DataFrame(final_energy_demand).sort_values(
        ["weather_year", "baseload_gw", "sector", "carrier"]
    )
    profile_table = pd.concat(
        [
            profile.assign(weather_year=year, baseload_gw=capacity)
            for year, (capacity, profile) in sorted(sector_profiles.items())
        ]
    ).reset_index().sort_values(["weather_year", "Time"])

    node_table = pd.concat(baseload_nodes, ignore_index=True).sort_values(
        ["weather_year", "baseload_gw", "country", "node"]
    )
    country_table = node_table.groupby(
        ["weather_year", "baseload_gw", "country"], as_index=False
    )[["installed_capacity_gw", "generation_twh"]].sum()
    country_table["capacity_share_percent"] = np.where(
        country_table.baseload_gw > 0,
        country_table.installed_capacity_gw / country_table.baseload_gw * 100,
        0.0,
    )
    country_table["full_load_hours"] = (
        country_table.generation_twh * 1e3
        / country_table.installed_capacity_gw.where(
            country_table.installed_capacity_gw > 1e-9
        )
    )

    asset_table = pd.DataFrame(heat_assets)
    if not asset_table.empty:
        keys = ["weather_year", "baseload_gw", "component", "carrier", "role", "cost_scope"]
        asset_table = asset_table.groupby(keys, as_index=False).sum(numeric_only=True)
        asset_table["direct_asset_cost_bn_eur"] = (
            asset_table.annualized_fixed_cost_bn_eur
            + asset_table.direct_variable_cost_bn_eur
        )
        asset_table = asset_table.sort_values(keys)
    load_table = pd.DataFrame(heat_loads)
    summary_table = pd.DataFrame(heat_summaries)
    if not summary_table.empty:
        summary_table["dedicated_heat_asset_direct_cost_bn_eur"] = (
            summary_table.dedicated_heat_asset_fixed_cost_bn_eur
            + summary_table.dedicated_heat_asset_variable_cost_bn_eur
        )
        summary_table = summary_table.sort_values(["weather_year", "baseload_gw"])
    return (costs, cost_detail_table, capacity_table, generation_table,
            energy_table, demand_table,
            profile_table,
            node_table, country_table,
            asset_table, load_table, summary_table)


def plot_costs(costs, output_file, case_steps):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for year, group in costs.groupby("weather_year"):
        # Leave visible gaps where an expected case has not been downloaded.
        group = group.set_index("baseload_gw").reindex(case_steps)
        axes[0].plot(
            group.index,
            group.system_cost_eur_per_year / 1e9,
            marker="o",
            label=str(year),
        )
        axes[1].plot(
            group.index,
            group.saving_vs_baseline_eur_per_year / 1e9,
            marker="o",
            label=str(year),
        )
    axes[0].set(ylabel="System cost (billion EUR/year)", title="Optimized system cost")
    axes[1].set(ylabel="Saving (billion EUR/year)", title="Relative to 0 GW baseline")
    for ax in axes:
        ax.set(xlabel="Added baseload capacity (GW)")
        ax.grid(alpha=0.25)
        ax.legend(title="Weather year")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def plot_system_cost_breakdown(cost_details, costs, output_dir):
    """Show cost groups plus generation, storage, and grid technologies."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Electricity generation": "#4477aa",
        "Heat supply": "#ee9944",
        "Fuel and resources": "#aa5533",
        "Conversion and industry": "#aa77bb",
        "Storage": "#66aa55",
        "Infrastructure": "#888888",
        "Carbon management": "#44aaaa",
        "Other objective terms": "#222222",
    }

    def stack(ax, values, palette):
        x = np.arange(len(values))
        positive = np.zeros(len(values))
        negative = np.zeros(len(values))
        for i, label in enumerate(values.columns):
            height = values[label].to_numpy(dtype=float)
            bottom = np.where(height >= 0, positive, negative)
            ax.bar(x, height, bottom=bottom, width=0.75,
                   label=label, color=palette(label, i))
            positive += np.maximum(height, 0)
            negative += np.minimum(height, 0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    for year, group in cost_details.groupby("weather_year"):
        cases = sorted(group.baseload_gw.unique())
        categories = group.pivot_table(
            index="baseload_gw", columns="category", values="total_cost_eur_per_year",
            aggfunc="sum", fill_value=0,
        ).reindex(index=cases, fill_value=0) / 1e9
        order = [name for name in colors if name in categories]
        categories = categories[order]

        generation = group.loc[group.category == "Electricity generation"]
        technologies = generation.pivot_table(
            index="baseload_gw", columns="carrier", values="total_cost_eur_per_year",
            aggfunc="sum", fill_value=0,
        ).reindex(index=cases, fill_value=0) / 1e9
        ranking = technologies.abs().max().sort_values(ascending=False)
        selected = ranking[ranking > 0.05].head(15).index
        other = technologies.drop(columns=selected).sum(axis=1)
        technologies = technologies[selected].copy()
        if other.abs().max() > 0.001:
            technologies["Other electricity generation"] = other

        facilities = group.loc[
            group.category.isin(["Storage", "Infrastructure"])
        ].copy()
        facilities["technology"] = facilities.category + ": " + facilities.carrier
        facility_values = facilities.pivot_table(
            index="baseload_gw", columns="technology", values="total_cost_eur_per_year",
            aggfunc="sum", fill_value=0,
        ).reindex(index=cases, fill_value=0) / 1e9
        facility_ranking = facility_values.abs().max().sort_values(ascending=False)
        facility_selected = facility_ranking[facility_ranking > 0.05].head(15).index
        facility_other = facility_values.drop(columns=facility_selected).sum(axis=1)
        facility_values = facility_values[facility_selected].copy()
        if facility_other.abs().max() > 0.001:
            facility_values["Other storage and infrastructure"] = facility_other

        fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
        stack(axes[0], categories, lambda name, _: colors[name])
        year_costs = costs.loc[costs.weather_year == year].set_index("baseload_gw")
        axes[0].plot(np.arange(len(cases)),
                     year_costs.loc[cases, "system_cost_eur_per_year"] / 1e9,
                     "ko", markersize=3, label="Reported objective")
        axes[0].set(title=f"{year}: annual system cost by function",
                    ylabel="Billion EUR/year")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

        if not technologies.empty:
            palette = plt.get_cmap("tab20")
            stack(axes[1], technologies,
                  lambda _, i: palette(i % palette.N))
            axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        axes[1].set(title="Electricity-generation cost by technology",
                    ylabel="Billion EUR/year")
        if not facility_values.empty:
            palette = plt.get_cmap("tab20")
            stack(axes[2], facility_values,
                  lambda _, i: palette(i % palette.N))
            axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        axes[2].set(title="Storage and infrastructure cost by technology",
                    ylabel="Billion EUR/year",
                    xlabel="Added baseload capacity (GW)")
        axes[2].set_xticks(np.arange(len(cases)), [f"{case:g}" for case in cases])
        fig.text(0.02, 0.01,
                 "Objective costs: new investment and operation; existing fixed assets' "
                 "capital costs are excluded. Full carrier detail is in system_cost_breakdown.csv.",
                 fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 0.78, 1))
        fig.savefig(output_dir / f"system_cost_breakdown_{year}.png", dpi=160)
        plt.close(fig)


def plot_capacity_changes(capacities, output_dir, case_steps, top_carriers=12):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (year, component, unit, basis), group in capacities.groupby(
        ["weather_year", "component", "unit", "capacity_basis"]
    ):
        if component == "Store":
            # Fuel and CO2 stores can be commodity accounting pools; keep them
            # in the CSV, but plot physical storage energy capacities only.
            group = group.loc[
                group.carrier.str.contains(
                    r"battery|water tanks|H2 Store", case=False, regex=True
                )
            ]
        ranking = (
            group.groupby("carrier").delta_vs_baseline
            .agg(lambda values: values.abs().max())
            .sort_values(ascending=False)
        )
        selected = ranking.head(top_carriers).index.tolist()
        if component == "Generator" and "baseload_candidate" in ranking:
            if "baseload_candidate" not in selected:
                selected.append("baseload_candidate")
        if not selected or not (ranking.loc[selected] > 0).any():
            continue

        values = group.pivot(
            index="carrier", columns="baseload_gw", values="delta_vs_baseline"
        ).reindex(index=selected, columns=case_steps)
        limit = values.abs().max().max()
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        fig, ax = plt.subplots(figsize=(9, max(4, len(selected) * 0.38 + 1.8)))
        image = ax.imshow(
            values.to_numpy(dtype=float),
            aspect="auto",
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
        )
        ax.set_xticks(range(len(case_steps)), case_steps)
        ax.set_yticks(range(len(selected)), selected)
        ax.set(xlabel="Added baseload capacity (GW)")
        ax.set_title(f"{year}: {component} capacity change vs 0 GW ({basis})")
        fig.colorbar(image, ax=ax, label=f"Change in installed capacity ({unit})")
        fig.tight_layout()
        fig.savefig(output_dir / f"capacity_change_{year}_{component.lower()}.png", dpi=160)
        plt.close(fig)


def plot_stacked_carriers(table, value, cases, title, ylabel, output_file, demand=None):
    import matplotlib.pyplot as plt

    if table.empty:
        return
    values = table.pivot_table(
        index="baseload_gw", columns="carrier", values=value,
        aggfunc="sum", fill_value=0,
    ).reindex(index=cases, fill_value=0)
    values = values.loc[:, values.max().sort_values(ascending=False).index]
    values = values.loc[:, values.max() > 1e-6]
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(cases))
    palette = plt.get_cmap("tab20")
    for i, carrier in enumerate(values):
        height = values[carrier].to_numpy()
        ax.bar(range(len(cases)), height, bottom=bottom,
               color=palette(i % 20), label=carrier)
        bottom += height
    if demand is not None:
        ax.plot(range(len(cases)), demand.reindex(cases), "ko--",
                linewidth=1.3, markersize=4, label="Heat load at district bus")
    ax.set_xticks(range(len(cases)), [f"{case:g}" for case in cases])
    ax.set(xlabel="Added baseload capacity (GW)", ylabel=ylabel, title=title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_electric_generation(generation, output_dir):
    import matplotlib.pyplot as plt

    for year, group in generation.groupby("weather_year"):
        cases = sorted(group.baseload_gw.unique())
        plot_stacked_carriers(
            group, "installed_capacity_gw_e", cases,
            f"{year}: installed electricity-generation capacity",
            "Installed electrical capacity (GW)",
            output_dir / f"generation_capacity_stacked_{year}.png",
        )
        values = group.pivot_table(
            index="carrier", columns="baseload_gw",
            values="installed_capacity_gw_e", aggfunc="sum", fill_value=0,
        ).reindex(columns=cases, fill_value=0)
        values = values.loc[values.max(axis=1).sort_values(ascending=False).index]
        values = values.loc[values.max(axis=1) > 1e-6]
        if values.empty:
            continue
        x = np.arange(len(values))
        width = 0.8 / len(cases)
        fig, ax = plt.subplots(figsize=(max(12, 0.7 * len(values) + 5), 5.5))
        for i, capacity in enumerate(cases):
            ax.bar(x + (i - (len(cases) - 1) / 2) * width,
                   values[capacity], width=width, label=f"{capacity:g} GW")
        ax.set_xticks(x, values.index, rotation=45, ha="right")
        ax.set(ylabel="Installed electrical capacity (GW)",
               title=f"{year}: generation technologies by baseload scenario")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(title="Added baseload")
        fig.tight_layout()
        fig.savefig(output_dir / f"generation_capacity_grouped_{year}.png", dpi=160)
        plt.close(fig)


def plot_electricity_generated(energy, output_dir):
    import matplotlib.pyplot as plt

    for year, group in energy.groupby("weather_year"):
        cases = sorted(group.baseload_gw.unique())
        values = group.pivot_table(
            index="carrier", columns="baseload_gw", values="generation_twh",
            aggfunc="sum", fill_value=0,
        ).reindex(columns=cases, fill_value=0)
        values = values.loc[values.max(axis=1).sort_values(ascending=False).index]
        values = values.loc[values.max(axis=1) >= 0.01]
        if values.empty:
            continue
        x = np.arange(len(values))
        width = 0.8 / len(cases)
        fig, ax = plt.subplots(figsize=(max(12, 0.8 * len(values) + 4), 5.5))
        for i, capacity in enumerate(cases):
            ax.bar(x + (i - (len(cases) - 1) / 2) * width,
                   values[capacity], width=width, label=f"{capacity:g} GW")
        ax.set_xticks(x, values.index, rotation=45, ha="right")
        ax.set(ylabel="Annual electricity supplied (TWh)",
               title=f"{year}: electricity output by technology",
               xlabel="Technology")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(title="Added baseload")
        fig.tight_layout()
        fig.savefig(output_dir / f"electricity_generation_{year}.png", dpi=160)
        plt.close(fig)


def plot_sector_final_energy_demand(demand, energy, output_dir):
    import matplotlib.pyplot as plt

    for year, group in demand.groupby("weather_year"):
        baseline = 0 if 0 in group.baseload_gw.values else group.baseload_gw.min()
        values = group.loc[group.baseload_gw == baseline].groupby("sector").demand_twh.sum()
        values = values.reindex(["Transport", "Heating", "Industry"], fill_value=0)
        baseload = energy.loc[
            (energy.weather_year == year) & (energy.carrier == "baseload_candidate")
        ].sort_values("baseload_gw")
        if not baseload.empty:
            largest = baseload.iloc[-1]
            values.loc[f"Baseload {largest.baseload_gw:g} GW\n(electricity supply)"] = (
                largest.generation_twh
            )

        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = ax.bar(values.index, values.to_numpy(),
                      color=["#4477aa", "#66aa55", "#cc8844", "#7a5195"][:len(values)])
        ax.axvline(2.5, color="0.6", linewidth=1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f"{value:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.set(ylabel="Annual energy (TWh)",
               title=f"{year}: sector final-energy demand vs baseload output")
        ax.set_ylim(0, max(values) * 1.12)
        ax.grid(axis="y", alpha=0.2)
        fig.text(0.02, 0.01, "Sector bars mix electricity, fuels and delivered heat; "
                 "the baseload bar is electricity only. Not directly substitutable.",
                 fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(output_dir / f"sector_final_energy_demand_{year}.png", dpi=160)
        plt.close(fig)


def plot_sector_load_profiles(profiles, output_dir):
    import matplotlib.pyplot as plt

    colors = {
        "Transport": "#4477aa",
        "Heating": "#66aa55",
        "Industry": "#cc8844",
        "Other electricity": "#7a5195",
    }
    for year, group in profiles.groupby("weather_year"):
        fig, ax = plt.subplots(figsize=(16, 5.5))
        for sector, color in colors.items():
            ax.plot(group.Time, group[sector], color=color, linewidth=0.75,
                    label=sector)
        capacity = group.baseload_gw.iloc[0]
        ax.set(
            title=f"{year}: Europe-wide load by sector ({capacity:g} GW baseload case)",
            xlabel="Time",
            ylabel="Load (GW)",
        )
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / f"sector_load_profile_{year}.png", dpi=160)
        plt.close(fig)


def plot_baseload_siting(nodes, countries, output_dir):
    import matplotlib.pyplot as plt

    for table, label, limit in ((countries, "countries", None), (nodes, "nodes", 30)):
        for year, group in table.groupby("weather_year"):
            if label == "nodes":
                group = group.copy()
                group["location"] = group.node + " (" + group.country + ")"
            else:
                group = group.rename(columns={"country": "location"})
            values = group.pivot_table(
                index="location", columns="baseload_gw",
                values="installed_capacity_gw", aggfunc="sum", fill_value=0,
            )
            values = values.reindex(columns=sorted(group.baseload_gw.unique()), fill_value=0)
            values = values.loc[values.max(axis=1).sort_values(ascending=False).index]
            values = values.loc[values.max(axis=1) >= 0.01]
            if limit is not None:
                values = values.head(limit)
            if values.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(7, len(values.columns) * 1.1),
                                            max(4, len(values) * 0.28 + 1.5)))
            image = ax.imshow(values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(values.columns)),
                          [f"{capacity:g}" for capacity in values.columns])
            ax.set_yticks(range(len(values)), values.index)
            ax.set(xlabel="Added baseload capacity (GW)",
                   title=f"{year}: experimental baseload capacity by {label}")
            if label == "countries":
                for row, capacities in enumerate(values.to_numpy()):
                    for col, capacity in enumerate(capacities):
                        if capacity >= 0.05:
                            ax.text(col, row, f"{capacity:.1f}", ha="center", va="center",
                                    color="black" if capacity > values.to_numpy().max() / 2
                                    else "white", fontsize=8)
            fig.colorbar(image, ax=ax, label="Installed capacity (GW)")
            fig.tight_layout()
            fig.savefig(output_dir / f"baseload_siting_{label}_{year}.png", dpi=160)
            plt.close(fig)


def plot_district_heating(assets, summaries, output_dir):
    import matplotlib.pyplot as plt

    if assets.empty or summaries.empty:
        return
    for year, group in assets.groupby("weather_year"):
        cases = sorted(group.baseload_gw.unique())
        supply = group.loc[
            group.role.isin(["heat_supply", "chp_heat", "process_heat"])
        ]
        demand = summaries.loc[
            summaries.weather_year == year
        ].set_index("baseload_gw").heat_at_bus_twh_th
        plot_stacked_carriers(
            supply, "heat_injected_twh_th", cases,
            f"{year}: heat injected into district heating",
            "Annual heat (TWh thermal)",
            output_dir / f"district_heating_generation_{year}.png",
            demand=demand,
        )
        plot_stacked_carriers(
            supply, "installed_heat_output_capacity_gw_th", cases,
            f"{year}: district-heat production capacity",
            "Installed heat output at mean efficiency (GW thermal)",
            output_dir / f"district_heating_capacity_{year}.png",
        )
        dedicated = group.loc[group.cost_scope == "district_heat_only"]
        plot_stacked_carriers(
            dedicated, "direct_asset_cost_bn_eur", cases,
            f"{year}: dedicated district-heat asset costs",
            "Annualized fixed + own variable cost (billion EUR/year)",
            output_dir / f"district_heating_direct_cost_{year}.png",
        )
        year_summary = summaries.loc[
            summaries.weather_year == year
        ].set_index("baseload_gw").reindex(cases)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(range(len(cases)), year_summary.thermal_storage_capacity_gwh_th)
        axes[0].set(ylabel="Thermal storage energy (GWh)", title="Water-tank capacity")
        axes[1].plot(range(len(cases)), year_summary.storage_charge_twh_th,
                     marker="o", label="Charging")
        axes[1].plot(range(len(cases)), year_summary.storage_discharge_twh_th,
                     marker="o", label="Discharging")
        axes[1].set(ylabel="Annual heat (TWh)", title="Storage operation")
        axes[1].legend()
        for ax in axes:
            ax.set_xticks(range(len(cases)), [f"{case:g}" for case in cases])
            ax.set(xlabel="Added baseload capacity (GW)")
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"district_heating_storage_{year}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Solved .nc files or directories containing them (searched recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseload_comparison"),
        help="Directory for comparison CSVs and cost plot",
    )
    parser.add_argument(
        "--expected-capacities",
        nargs="+",
        type=float,
        help="Expected GW cases; show gaps for missing networks (default: loaded cases)",
    )
    args = parser.parse_args()

    (costs, cost_details, capacities, generation, energy, demand, profiles,
     baseload_nodes, baseload_countries,
     heat_assets, heat_loads, heat_summary) = compare(find_networks(args.inputs))
    case_steps = (
        sorted(set(args.expected_capacities))
        if args.expected_capacities is not None
        else sorted(costs.baseload_gw.unique())
    )
    for year, group in costs.groupby("weather_year"):
        missing = sorted(set(case_steps) - set(group.baseload_gw))
        if missing:
            print(f"Weather {year}: missing baseload cases {missing} GW")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    costs.to_csv(args.output_dir / "system_costs.csv", index=False)
    cost_details.to_csv(args.output_dir / "system_cost_breakdown.csv", index=False)
    capacities.to_csv(args.output_dir / "installed_capacities.csv", index=False)
    generation.to_csv(args.output_dir / "generation_capacities.csv", index=False)
    energy.to_csv(args.output_dir / "electricity_generation_twh.csv", index=False)
    demand.to_csv(args.output_dir / "sector_final_energy_demand_twh.csv", index=False)
    profiles.to_csv(args.output_dir / "sector_load_profile_gw.csv", index=False)
    (args.output_dir / "sector_final_energy_demand_notes.txt").write_text(
        "Annual demand uses snapshot-weighted solved dispatch in TWh.\n"
        "Transport includes land-transport oil, aviation kerosene, shipping oil "
        "and methanol Loads, plus BEV charger electricity net of V2G output.\n"
        "Heating includes fuels and electricity entering rural and urban-decentral "
        "heat pumps, boilers and resistive heaters, solar thermal output, and "
        "district heat delivered after removing the model's distribution-loss markup. "
        "Ambient heat captured by heat pumps is excluded.\n"
        "Industry includes electricity, hydrogen, gas, coal, solid biomass and "
        "low-temperature heat Loads. Naphtha and methanol feedstocks are excluded; "
        "industrial hydrogen may still include feedstock use.\n"
        "Baseload output is electricity and cannot replace all sector carriers "
        "one-for-one. These are approximate final-energy quantities, not a "
        "formal statistical energy balance.\n"
        "The 3-hour load profile sums these sectors across all model buses "
        "for the 0 GW case (or the lowest available case). Its additional "
        "Other electricity line contains direct Loads with carrier electricity; "
        "all lines are snapshot power in GW.\n",
        encoding="utf-8",
    )
    baseload_nodes.to_csv(args.output_dir / "baseload_siting_nodes.csv", index=False)
    baseload_countries.to_csv(args.output_dir / "baseload_siting_countries.csv", index=False)
    heat_assets.to_csv(args.output_dir / "district_heating_technologies.csv", index=False)
    heat_loads.to_csv(args.output_dir / "district_heating_loads.csv", index=False)
    heat_summary.to_csv(args.output_dir / "district_heating_summary.csv", index=False)
    plot_costs(costs, args.output_dir / "system_costs.png", case_steps)
    plot_system_cost_breakdown(cost_details, costs, args.output_dir)
    plot_capacity_changes(capacities, args.output_dir, case_steps)
    plot_electric_generation(generation, args.output_dir)
    plot_electricity_generated(energy, args.output_dir)
    plot_sector_final_energy_demand(demand, energy, args.output_dir)
    plot_sector_load_profiles(profiles, args.output_dir)
    plot_baseload_siting(baseload_nodes, baseload_countries, args.output_dir)
    plot_district_heating(heat_assets, heat_summary, args.output_dir)
    (args.output_dir / "district_heating_notes.txt").write_text(
        "District heating means buses with carrier 'urban central heat'.\n"
        "Heat at bus includes every load attached to those buses, including low-temperature "
        "industrial heat. Estimated delivered heat removes the configured distribution "
        "loss from the urban-central-heat load only.\n"
        "Heat injected by CHP and shared processes is included in generation. Storage "
        "discharge is reported separately to avoid counting stored heat as new production.\n"
        "Link input capacity is p_nom at bus0. Heat output capacity is p_nom times the "
        "time-weighted mean efficiency at the district-heat port; heat-pump output "
        "capacity therefore varies with the weather profile. Store capacity is energy.\n"
        "Technology costs are full annualized fixed cost plus each asset's own marginal "
        "cost. CHP and industrial-process assets also serve other sectors, so their full "
        "costs are shown but excluded from the dedicated-heat cost total. Electricity, "
        "fuel and emissions costs incurred elsewhere are not allocated to heat. This is "
        "not the total economic cost of district heating. The model has no separate "
        "district-heating distribution network investment; it represents distribution "
        "losses in the urban-central-heat load.\n",
        encoding="utf-8",
    )
    print(f"Wrote comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
