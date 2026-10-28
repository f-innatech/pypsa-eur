# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Check heat extraction in one solved baseload CHP network."""

import argparse
from pathlib import Path

import numpy as np
import pypsa

from baseload_experiment import CARRIER, HEAT_CARRIER, heat_coupling_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path, help="Solved CHP .nc network")
    args = parser.parse_args()

    n = pypsa.Network(args.network)
    options = n.meta.get("baseload_experiment", {}).get("district_heat", {})
    if not options.get("enabled", False):
        parser.error("This network was not solved with CHP enabled")
    penalty, heat_limit = heat_coupling_parameters(options)
    ratio = penalty * heat_limit

    generators = n.generators.index[n.generators.carrier == CARRIER]
    links = n.links.index[n.links.carrier == HEAT_CARRIER]
    if generators.empty or links.empty:
        parser.error("No baseload generators or district-heat links in this network")

    requested_gw = float(n.meta["wildcards"]["capacity_gw"])
    built_gw = n.generators.loc[generators, "p_nom_opt"].sum() / 1e3
    if abs(built_gw - requested_gw) > 1e-3:
        raise ValueError(f"Built {built_gw:.3f} GW; requested {requested_gw:.3f} GW")

    tolerance_mw = 0.1
    total_heat_mwh = 0.0
    total_loss_mwh = 0.0
    maximum_heat_ratio = 0.0
    weights = n.snapshot_weightings.generators
    for link in links:
        ac_bus = n.links.at[link, "bus0"]
        generator = f"{ac_bus} {CARRIER}"
        if generator not in generators:
            raise ValueError(f"No colocated baseload generator for {link}")
        if n.links.at[link, "bus1"] != f"{ac_bus} urban central heat":
            raise ValueError(f"Wrong district-heating bus for {link}")

        gross_mw = n.generators_t.p[generator]
        electricity_lost_mw = n.links_t.p0[link]
        heat_mw = -n.links_t.p1[link]
        if not np.isfinite(gross_mw).all() or not np.isfinite(electricity_lost_mw).all() or not np.isfinite(heat_mw).all():
            raise ValueError(f"Non-finite CHP dispatch at {link}")
        capacity_error_mw = abs(
            n.links.at[link, "p_nom_opt"]
            - ratio * n.generators.at[generator, "p_nom_opt"]
        )
        if not np.isfinite(capacity_error_mw) or capacity_error_mw > tolerance_mw:
            raise ValueError(f"Heat-link capacity is not coupled at {link}")
        if (electricity_lost_mw - ratio * gross_mw).max() > tolerance_mw:
            raise ValueError(f"Heat extraction exceeds generator output at {link}")
        if (heat_mw - electricity_lost_mw / penalty).abs().max() > tolerance_mw:
            raise ValueError(f"Heat/electricity-loss ratio is wrong at {link}")
        if electricity_lost_mw.min() < -tolerance_mw:
            raise ValueError(f"Heat link reverses electricity flow at {link}")

        total_heat_mwh += (heat_mw * weights).sum()
        total_loss_mwh += (electricity_lost_mw * weights).sum()
        active = gross_mw > tolerance_mw
        if active.any():
            maximum_heat_ratio = max(
                maximum_heat_ratio,
                (heat_mw[active] / gross_mw[active]).max(),
            )

    gross_mwh = n.generators_t.p[generators].mul(weights, axis=0).sum().sum()
    print(f"Baseload capacity: {built_gw:.1f} GW")
    print(f"CHP links: {len(links)}")
    print(f"Gross electricity: {gross_mwh / 1e6:.3f} TWh/year")
    print(f"District heat: {total_heat_mwh / 1e6:.3f} TWh/year")
    print(f"Electricity foregone: {total_loss_mwh / 1e6:.3f} TWh/year")
    print(f"Net electricity: {(gross_mwh - total_loss_mwh) / 1e6:.3f} TWh/year")
    print(f"Electricity loss per heat: {penalty:.3f} MWh/MWh")
    print(f"Maximum heat/gross-electricity ratio: {maximum_heat_ratio:.3f} (limit {heat_limit:.3f})")
    if total_heat_mwh <= tolerance_mw:
        print("CHP constraints pass, but the optimizer did not use district heat.")
    else:
        print("CHP dispatch and capacity checks passed.")


if __name__ == "__main__":
    main()
