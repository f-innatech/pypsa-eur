# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Collect annual cost and capacity value for all baseload weather cases."""

from pathlib import Path

import pandas as pd


def summarize(files):
    results = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    if results.duplicated(["scenario", "baseload_gw"]).any():
        raise ValueError("Duplicate baseload capacity case in sweep results")

    results = results.sort_values(["weather_year", "baseload_gw"]).reset_index(
        drop=True
    )
    for scenario, group in results.groupby("scenario"):
        if (group["baseload_gw"] == 0).sum() != 1:
            raise ValueError(f"Scenario {scenario} has no unique 0 GW baseline")

    baseline = results.loc[results.baseload_gw == 0].set_index("scenario")
    results["baseline_cost_eur_per_year"] = results.scenario.map(
        baseline.system_cost_eur_per_year
    )
    results["cost_saving_eur_per_year"] = (
        results.baseline_cost_eur_per_year - results.system_cost_eur_per_year
    )
    results["average_value_eur_per_mw_year"] = (
        results.cost_saving_eur_per_year / (results.baseload_gw * 1e3)
    )

    groups = results.groupby("scenario", sort=False)
    previous_cost = groups.system_cost_eur_per_year.shift()
    previous_capacity = groups.baseload_gw.shift()
    results["marginal_value_eur_per_mw_year"] = (
        (previous_cost - results.system_cost_eur_per_year)
        / ((results.baseload_gw - previous_capacity) * 1e3)
    )
    return results


if __name__ == "__main__":
    results = summarize(snakemake.input.metrics)  # noqa: F821
    output = Path(snakemake.output[0])  # noqa: F821
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
