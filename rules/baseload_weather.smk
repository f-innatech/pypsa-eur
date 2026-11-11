# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

capacities_gw = config["baseload_experiment"]["capacities_gw"]
baseload_output_dir = config["baseload_experiment"].get("output_dir", "baseload")
if (
    not capacities_gw
    or capacities_gw[0] != 0
    or any(type(capacity) is not int or capacity < 0 for capacity in capacities_gw)
    or capacities_gw != sorted(set(capacities_gw))
):
    raise ValueError("Baseload capacities must be sorted, unique, non-negative integers starting at 0")
capacity_wildcard = "|".join(str(capacity) for capacity in sorted(capacities_gw, reverse=True))
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
        capacity_gw=capacity_wildcard,
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


if config.get("baseload_cost_test", {}).get("enabled", False):
    cost_test = config["baseload_cost_test"]
    cost_test_dir = cost_test["output_dir"]
    cost_cases = cost_test["capex_shares"]
    if not cost_cases or any(
        not 0 < float(share) < 1 for share in cost_cases.values()
    ):
        raise ValueError("CAPEX shares must be strictly between 0 and 1")
    cost_case_wildcard = "|".join(cost_cases)

    rule solve_baseload_cost_case:
        wildcard_constraints:
            run="weather-2010",
            cost_case=cost_case_wildcard,
        params:
            solving=config_provider("solving"),
            foresight=config_provider("foresight"),
            co2_sequestration_potential=config_provider(
                "sector", "co2_sequestration_potential", default=200
            ),
            custom_extra_functionality=input_custom_extra_functionality,
            baseload_capacity_gw=cost_test["reference_capacity_gw"],
            baseload_capacity_mode="endogenous",
            baseload_cost_share=lambda w: cost_cases[w.cost_case],
            baseload_district_heat=config["baseload_experiment"].get("district_heat", {}),
        input:
            network=resources(
                "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
            ),
            baseload_costs=lambda w: resources(
                f"costs_{config_provider('costs', 'year')(w)}.csv"
            ),
            baseload_availability=config["baseload_experiment"]["availability_file"],
            baseload_reference_baseline=cost_test["reference_baseline_metrics"],
            baseload_reference_free=cost_test["reference_free_metrics"],
        output:
            network=RESULTS
            + cost_test_dir
            + "/networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_cost_{cost_case}.nc",
            config=RESULTS
            + cost_test_dir
            + "/configs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_cost_{cost_case}.yaml",
            metrics=RESULTS
            + cost_test_dir
            + "/metrics/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_cost_{cost_case}.csv",
        shadow:
            shadow_config
        log:
            solver=RESULTS
            + cost_test_dir
            + "/logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_cost_{cost_case}_solver.log",
            memory=RESULTS
            + cost_test_dir
            + "/logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_cost_{cost_case}_memory.log",
            python=RESULTS
            + cost_test_dir
            + "/logs/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_cost_{cost_case}_python.log",
        threads: solver_threads
        resources:
            mem_mb=config_provider("solving", "mem_mb"),
            runtime=config_provider("solving", "runtime", default="6h"),
        benchmark:
            RESULTS
            + cost_test_dir
            + "/benchmarks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_cost_{cost_case}"
        conda:
            "../envs/environment.yaml"
        script:
            "../scripts/solve_network.py"
