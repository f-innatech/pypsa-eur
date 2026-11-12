#!/usr/bin/env bash
#SBATCH --job-name=baseload-mustrun-2010
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=baseload-mustrun-2010-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pypsa-eur repository root}"

if [[ ! -f Snakefile || ! -f config/config.baseload-capex-opex-2010.yaml ]]; then
    echo "Submit from the pypsa-eur repository root with the cost-test config." >&2
    exit 1
fi

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) cost_case=capex80 ;;
    1) cost_case=capex20 ;;
    *) echo "Expected SLURM_ARRAY_TASK_ID between 0 and 1." >&2; exit 1 ;;
esac

for input_file in \
    resources/weather-2010/networks/base_s_128___2050.nc \
    resources/weather-2010/costs_2050.csv \
    results/weather-2010/baseload-chp-free-opex/metrics/base_s_128___2050_b0.csv \
    results/weather-2010/baseload-chp-free-opex/metrics/base_s_128___2050_b300.csv; do
    if [[ ! -r "$input_file" ]]; then
        echo "Missing input: $input_file" >&2
        exit 1
    fi
done

if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load devel/miniforge
fi
if ! command -v conda >/dev/null 2>&1; then
    for conda_init in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh"; do
        if [[ -f "$conda_init" ]]; then
            # shellcheck disable=SC1090
            source "$conda_init"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Load the cluster's Conda module." >&2
    exit 1
fi
unset PYTHONPATH PYTHONHOME
eval "$(conda shell.bash hook)"
conda activate pypsa-eur

export GRB_LICENSE_FILE=/home/fr/fr_fr/fr_fp1059/opt/gurobi/gurobi.lic
if [[ ! -r "$GRB_LICENSE_FILE" ]]; then
    echo "Gurobi license is not readable: $GRB_LICENSE_FILE" >&2
    exit 1
fi

target="results/weather-2010/baseload-chp-cost-test-must-run/metrics/base_s_128___2050_cost_${cost_case}.csv"
echo "Solving $target with 16 Gurobi threads on ${SLURM_JOB_NODELIST:-unknown node}"

snakemake --snakefile Snakefile \
    --configfile \
        config/config.baseload-weather.yaml \
        config/config.baseload-2050.yaml \
        config/config.baseload-chp.yaml \
        config/config.baseload-offline.yaml \
        config/config.baseload-16threads.yaml \
        config/config.baseload-capex-opex-2010.yaml \
    --allowed-rules solve_baseload_cost_case \
    --executor local \
    --cores 16 \
    --resources mem_mb=120000 \
    --set-threads solve_baseload_cost_case=16 \
    --latency-wait 60 \
    --rerun-incomplete \
    "$target"
