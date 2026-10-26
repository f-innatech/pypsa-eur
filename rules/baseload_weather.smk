# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

capacities_gw = config["baseload_experiment"]["capacities_gw"]
baseload_output_dir = config["baseload_experiment"].get("output_dir", "baseload")
if capacities_gw != list(range(0, 61, 10)):
    raise ValueError("The baseload sweep expects 0, 10, ..., 60 GW")
for key in ("clusters", "opts", "sector_opts", "planning_horizons"):
    if len(config["scenario"][key]) != 1:
        raise ValueError(f"The baseload sweep requires one scenario.{key} value")

BASELOAD_NETWORK = (
    RESULTS
    + baseload_output_dir
    + "/networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_b{capacity_gw}.nc"
)
BASELOAD_METRICS = (
    RESULTS
    + baseload_output_dir
    + "/metrics/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_b{capacity_gw}.csv"
)


def baseload_case_path(pattern, run_name, capacity_gw):
    return pattern.format(
        run=run_name,
        capacity_gw=capacity_gw,
        **{key: values[0] for key, values in config["scenario"].items()},
    )


BASELOAD_CASES = [
    (run_name, capacity_gw)
    for run_name in config["run"]["name"]
    for capacity_gw in capacities_gw
]


rule baseload_sweep:
    input:
        config["baseload_experiment"]["summary"],


rule summarize_baseload_sweep:
    input:
        metrics=[
            baseload_case_path(BASELOAD_METRICS, run_name, capacity_gw)
            for run_name, capacity_gw in BASELOAD_CASES
        ],
    output:
        config["baseload_experiment"]["summary"],
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/summarize_baseload_sweep.py"


rule solve_baseload_case:
    wildcard_constraints:
        capacity_gw="0|10|20|30|40|50|60",
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        baseload_capacity_gw=lambda w: int(w.capacity_gw),
        baseload_district_heat=config["baseload_experiment"].get("district_heat", {}),
    input:
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
        ),
        baseload_costs=lambda w: resources(
            f"costs_{config_provider('costs', 'year')(w)}.csv"
        ),
        baseload_availability=config["baseload_experiment"]["availability_file"],
    output:
        network=BASELOAD_NETWORK,
        config=RESULTS
        + baseload_output_dir
        + "/configs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_b{capacity_gw}.yaml",
        metrics=BASELOAD_METRICS,
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + baseload_output_dir
        + "/logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_b{capacity_gw}_solver.log",
        memory=RESULTS
        + baseload_output_dir
        + "/logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_b{capacity_gw}_memory.log",
        python=RESULTS
        + baseload_output_dir
        + "/logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_b{capacity_gw}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        RESULTS
        + baseload_output_dir
        + "/benchmarks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_b{capacity_gw}"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"
