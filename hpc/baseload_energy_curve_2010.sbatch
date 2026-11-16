#!/usr/bin/env bash
#SBATCH --job-name=baseload-energy-2010
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --array=1-9%2
#SBATCH --output=baseload-energy-2010-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pypsa-eur repository root}"

if [[ ! -f Snakefile || ! -f config/config.baseload-energy-curve-2010.yaml ]]; then
    echo "Submit from the pypsa-eur repository root." >&2
    exit 1
fi

cf_percent=$((SLURM_ARRAY_TASK_ID * 10))
case "$cf_percent" in
    10|20|30|40|50|60|70|80|90) ;;
    *) echo "Expected array task ID from 1 to 9." >&2; exit 1 ;;
esac

for input_file in \
    resources/weather-2010/networks/base_s_128___2050.nc \
    resources/weather-2010/costs_2050.csv \
    data/nuclear_p_max_pu.csv \
    results/weather-2010/baseload-chp-free-opex/metrics/base_s_128___2050_b0.csv; do
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

target="results/weather-2010/baseload-chp-energy-curve/metrics/base_s_128___2050_b300_cf${cf_percent}.csv"
echo "Solving fixed 300 GW, target CF ${cf_percent}%, on ${SLURM_JOB_NODELIST:-unknown node}"

# Array tasks write distinct outputs; disable Snakemake's repository-wide lock.
snakemake --snakefile Snakefile \
    --configfile \
        config/config.baseload-weather.yaml \
        config/config.baseload-2050.yaml \
        config/config.baseload-chp.yaml \
        config/config.baseload-offline.yaml \
        config/config.baseload-16threads.yaml \
        config/config.baseload-energy-curve-2010.yaml \
    --allowed-rules solve_baseload_energy_case \
    --executor local \
    --cores 16 \
    --resources mem_mb=120000 \
    --set-threads solve_baseload_energy_case=16 \
    --latency-wait 60 \
    --rerun-incomplete \
    --nolock \
    "$target" \
    --rerun-triggers mtime
