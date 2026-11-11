"""Compare three-hourly electricity-demand shadow prices across weather years.

The plotted price is the load-weighted marginal price at low-voltage buses for
direct electricity loads. It is the system-cost saving from reducing 1 MWh of
those loads, distributed proportionally to their positive demand in that snapshot.
"""

import argparse
import gc
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa


ELECTRICITY_LOAD_CARRIERS = {
    "electricity",
    "industry electricity",
    "agriculture electricity",
}
CASE_IN_FILENAME = re.compile(r"_b(?P<capacity>0|300)\.nc$")


def shadow_price_profile(path):
    case = CASE_IN_FILENAME.search(path.name)
    if case is None:
        raise ValueError(f"Expected a 0 or 300 GW network: {path}")
    capacity_gw = int(case.group("capacity"))
    network = pypsa.Network(path)
    snapshots = pd.DatetimeIndex(network.snapshots)
    year = snapshots[0].year
    if (
        len(snapshots) != 2920
        or snapshots[0] != pd.Timestamp(year, 1, 1)
        or snapshots[-1] != pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(hours=3)
        or not (snapshots.to_series().diff().dropna() == pd.Timedelta(hours=3)).all()
    ):
        raise ValueError(f"Expected one complete, three-hourly weather year: {path}")

    bus_carriers = network.loads.bus.map(network.buses.carrier)
    load_names = network.loads.index[
        network.loads.carrier.isin(ELECTRICITY_LOAD_CARRIERS)
        & bus_carriers.eq("low voltage")
    ]
    if load_names.empty:
        raise ValueError(f"No direct electricity loads on low-voltage buses: {path}")

    load_power = network.loads_t.p.reindex(index=snapshots, columns=load_names)
    if load_power.isna().any().any():
        raise ValueError(f"Missing direct electricity load: {path}")
    # Negative Load entries are injections, not demand that can be reduced.
    load_power = load_power.clip(lower=0)
    bus_load = load_power.T.groupby(network.loads.loc[load_names, "bus"]).sum().T
    prices = network.buses_t.marginal_price.reindex(
        index=snapshots, columns=bus_load.columns
    )
    if prices.isna().any().any():
        raise ValueError(f"Missing low-voltage marginal prices: {path}")
    total_load = bus_load.sum(axis=1)
    if (total_load <= 0).any():
        raise ValueError(f"Zero direct electricity demand in a snapshot: {path}")

    profile = pd.DataFrame(
        {
            "hour_of_year": (snapshots - pd.Timestamp(year, 1, 1)).total_seconds() / 3600,
            "shadow_price_eur_per_mwh": (prices * bus_load).sum(axis=1) / total_load,
        }
    )
    del network
    gc.collect()
    return (year, capacity_gw), profile


def plot_winter_period(profiles, years, capacities, start, end, output_file):
    first_day = pd.Timestamp("2010-01-01")
    start_day = (start - first_day).days
    end_day = (end - first_day).days
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for year, color in zip(years, ("#c44e52", "#4c72b0")):
        for capacity in capacities:
            profile = profiles[year, capacity]
            days = profile.hour_of_year / 24
            selected = days.ge(start_day) & days.lt(end_day)
            ax.plot(
                days[selected],
                profile.shadow_price_eur_per_mwh[selected],
                color=color,
                linestyle="-" if capacity == 0 else "--",
                linewidth=0.8,
                alpha=0.8,
                label=f"{year}, {capacity} GW",
            )
    ticks = pd.date_range(start, end, freq="14D", inclusive="left")
    ax.set_xticks((ticks - first_day).days, ticks.strftime("%d %b"), rotation=35)
    ax.set_xlim(start_day, end_day)
    ax.set(
        title=f"{start:%B}–{end - pd.Timedelta(days=1):%B}: electricity-demand shadow price",
        xlabel="Date (weather years aligned by calendar date)",
        ylabel="Demand-weighted marginal price (EUR/MWh)",
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(alpha=0.2)
    ax.legend(title="Weather year, baseload capacity", ncol=2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "networks", nargs="+", type=Path,
        help="0 and/or 300 GW solved networks for two weather years",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseload_shadow_prices"))
    args = parser.parse_args()

    if len(args.networks) not in (2, 4):
        parser.error("Provide one capacity for two years, or both capacities for both years")
    profiles = dict(shadow_price_profile(path) for path in args.networks)
    years = sorted({year for year, _ in profiles})
    capacities = sorted({capacity for _, capacity in profiles})
    if len(years) != 2 or len(profiles) != len(args.networks) or len(profiles) != 2 * len(capacities):
        parser.error("Provide each capacity once for each of two different weather years")
    hours_by_case = [profile.hour_of_year.to_numpy() for profile in profiles.values()]
    if not all(np.array_equal(hours_by_case[0], hours) for hours in hours_by_case[1:]):
        parser.error("The weather years must have matching three-hourly calendars")

    output = pd.DataFrame({"hour_of_year": hours_by_case[0]})
    for year in years:
        for capacity in capacities:
            case_label = f"_b{capacity}" if len(capacities) > 1 else ""
            output[f"shadow_price_{year}{case_label}_eur_per_mwh"] = (
                profiles[year, capacity].shadow_price_eur_per_mwh.to_numpy()
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    capacity_label = "_".join(str(capacity) for capacity in capacities) + "gw"
    stem = f"electricity_shadow_prices_{capacity_label}"
    output.to_csv(args.output_dir / f"{stem}.csv", index=False)

    for label, start, end in (
        ("jan_mar", pd.Timestamp("2010-01-01"), pd.Timestamp("2010-04-01")),
        ("oct_dec", pd.Timestamp("2010-10-01"), pd.Timestamp("2011-01-01")),
    ):
        plot_winter_period(
            profiles, years, capacities, start, end,
            args.output_dir / f"{stem}_{label}.png",
        )
    print(f"Wrote two winter shadow-price plots and CSV to {args.output_dir}")


if __name__ == "__main__":
    main()
