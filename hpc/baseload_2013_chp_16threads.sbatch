#!/usr/bin/env bash
#SBATCH --job-name=baseload-2013-chp
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-4%2
#SBATCH --output=baseload-2013-chp-%A_%a.out

# Submit this array from the pypsa-eur repository root. Each task has distinct outputs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pypsa-eur repository root}"

if [[ ! -f Snakefile || ! -f config/config.baseload-16threads.yaml ]]; then
    echo "Submit from the pypsa-eur repository root." >&2
    exit 1
fi

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) capacity_gw=60 ;;
    1) capacity_gw=120 ;;
    2) capacity_gw=180 ;;
    3) capacity_gw=240 ;;
    4) capacity_gw=300 ;;
    *)
        echo "Expected SLURM_ARRAY_TASK_ID between 0 and 4." >&2
        exit 1
        ;;
esac

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

target="results/weather-2013/baseload-chp-free-opex/metrics/base_s_128___2050_b${capacity_gw}.csv"
echo "Solving ${target} with 16 Gurobi threads on ${SLURM_JOB_NODELIST:-unknown node}"

snakemake --snakefile Snakefile \
    --configfile \
        config/config.baseload-weather.yaml \
        config/config.baseload-2050.yaml \
        config/config.baseload-chp.yaml \
        config/config.baseload-offline.yaml \
        config/config.baseload-16threads.yaml \
    --allowed-rules solve_baseload_case \
    --executor local \
    --cores 16 \
    --resources mem_mb=120000 \
    --set-threads solve_baseload_case=16 \
    --latency-wait 60 \
    --rerun-incomplete \
    "$target"
