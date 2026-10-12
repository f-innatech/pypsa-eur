# 2045 baseload value sweep

Run from the repository root with the PyPSA-Eur environment and a configured
Snakemake cluster profile:

```bash
snakemake --snakefile Snakefile \
  --configfile config/config.baseload-weather.yaml \
  --software-deployment-method conda \
  --profile YOUR_CLUSTER_PROFILE \
  --jobs 21 baseload_sweep
```

To inspect the job graph first, replace the profile and `--jobs 21` arguments
with `--dry-run`. The target builds missing inputs, prepares three sector
networks, solves 21 cases, and writes
`results/baseload-weather/summary.csv`. Individual solved networks, configs,
metrics, logs, and benchmarks are under `results/weather-<year>/baseload/`.
Restarting the same command resumes incomplete cases.

The cases use the 2010, 2013, and 2018 weather years; 2045 technology costs;
the 2045 CO₂ limit of 5% of 1990 emissions; 128 spatial clusters; 3-hour
sector snapshots; and the standard sector settings with vehicle-to-grid on.
Reported electricity demand uses the 2013 profile in each weather year.
Weather-dependent renewable and heat profiles still change by year.

Each weather year has its own 0 GW baseline. Cases with 10–60 GW add
zero-construction-cost generators in 10 GW steps. Their siting is optimized
across AC buses, and their total nameplate capacity is fixed at the requested
step. They have 2045 nuclear fuel, variable and fixed operating costs, and
country-specific nuclear availability factors. They can dispatch between zero
and available capacity; no minimum generation or unit commitment is imposed.
The summary reports annual system cost, savings against that weather year's
baseline, and average and incremental value per MW-year. These values are net
of the candidate generators' operating costs and exclude their construction
costs. The absolute cost column is the PyPSA optimization objective; compare
cases within a weather year to measure savings.

The 2010 and 2018 atlite cutouts must be generated or placed at
`cutouts/weather-<year>/europe-<year>-sarah3-era5.nc`; the same applies to 2013
if it has not been built for this run. Building them requires access to ERA5
and SARAH3 data. The 2045 cost CSV must be retrievable or staged at
`resources/weather-<year>/costs_2045.csv` for each year; the standard
PyPSA-Eur data inputs must also be available. The solver defaults to Gurobi;
provide a working cluster licence or override the solver in the config. The
experiment requests 120 GB and 24 hours per solve; adjust these to the cluster
after a first run.
