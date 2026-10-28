#!/usr/bin/env bash
#SBATCH --job-name=baseload-sweep
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=baseload-controller-%j.out

# Submit this file from the pypsa-eur repository root.
# The small allocation runs Snakemake; Snakemake submits compute jobs to SLURM.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pypsa-eur repository root}"

if [[ ! -f Snakefile || ! -f config/config.baseload-weather.yaml ]]; then
    echo "Submit from the pypsa-eur repository root." >&2
    exit 1
fi

# bwUniCluster 3.0 provides Conda through this module.
if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load devel/miniforge
fi

# Also support an existing user Conda installation.
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
    echo "Conda was not found. Load the cluster's Conda module in this script." >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate pypsa-eur

export GRB_LICENSE_FILE=/home/fr/fr_fr/fr_fp1059/opt/gurobi/gurobi.lic
if [[ ! -r "$GRB_LICENSE_FILE" ]]; then
    echo "Gurobi license is not readable: $GRB_LICENSE_FILE" >&2
    exit 1
fi
python -c 'import snakemake_executor_plugin_slurm, gurobipy'

if [[ ! -f data/WDPA.gpkg || ! -f data/WDPA_WDOECM_marine.gpkg ]]; then
    echo "Warning: WDPA inputs are absent; this PyPSA-Eur version may fail to retrieve them." >&2
fi

config_files=(config/config.baseload-weather.yaml)
planning_year="${BASELOAD_YEAR:-2045}"
case "$planning_year" in
    2045) ;;
    2050) config_files+=(config/config.baseload-2050.yaml) ;;
    *)
        echo "BASELOAD_YEAR must be 2045 or 2050." >&2
        exit 1
        ;;
esac
result_subdir=baseload-free-opex
case "${BASELOAD_VARIANT:-chp}" in
    electricity) ;;
    chp)
        config_files+=(config/config.baseload-chp.yaml)
        result_subdir=baseload-chp-free-opex
        ;;
    *)
        echo "BASELOAD_VARIANT must be electricity or chp." >&2
        exit 1
        ;;
esac

case "${BASELOAD_MODE:-pilot}" in
    pilot)
        pilot_gw=0
        if [[ "${BASELOAD_VARIANT:-chp}" == chp ]]; then
            pilot_gw=60
        fi
        targets=("results/weather-2010/${result_subdir}/metrics/base_s_128___${planning_year}_b${pilot_gw}.csv")
        ;;
    full)
        targets=(baseload_sweep)
        ;;
    weather-2010)
        case "${BASELOAD_BATCH:-}" in
            0|1|2) ;;
            *)
                echo "Set BASELOAD_BATCH to 0, 1, or 2 for weather-2010." >&2
                exit 1
                ;;
        esac
        first_gw=$((BASELOAD_BATCH * 120))
        targets=(
            "results/weather-2010/${result_subdir}/metrics/base_s_128___${planning_year}_b${first_gw}.csv"
            "results/weather-2010/${result_subdir}/metrics/base_s_128___${planning_year}_b$((first_gw + 60)).csv"
        )
        ;;
    *)
        echo "BASELOAD_MODE must be pilot, full, or weather-2010." >&2
        exit 1
        ;;
esac

# Optional: export BASELOAD_PARTITION and/or BASELOAD_ACCOUNT before sbatch.
default_resources=(mem_mb=32000 runtime=1440)
if [[ -n "${BASELOAD_PARTITION:-}" ]]; then
    default_resources+=("slurm_partition=${BASELOAD_PARTITION}")
fi
if [[ -n "${BASELOAD_ACCOUNT:-}" ]]; then
    default_resources+=("slurm_account=${BASELOAD_ACCOUNT}")
fi

echo "Year: ${planning_year}; variant: ${BASELOAD_VARIANT:-chp}; mode: ${BASELOAD_MODE:-pilot}; targets: ${targets[*]}"
echo "Repository: $PWD; environment: $CONDA_DEFAULT_ENV"

snakemake --snakefile Snakefile \
    --configfile "${config_files[@]}" config/config.baseload-offline.yaml \
    --allowed-rules baseload_sweep summarize_baseload_sweep solve_baseload_case \
    --executor slurm \
    --jobs 2 \
    --local-cores 1 \
    --default-resources "${default_resources[@]}" \
    --set-resources \
        build_cutout:mem_mb=32000 \
        prepare_sector_network:mem_mb=64000 \
        solve_sector_network:runtime=2880 \
        solve_baseload_case:runtime=2880 \
    --latency-wait 60 \
    --rerun-incomplete \
    "${targets[@]}" \
    --rerun-triggers mtime

if [[ "${BASELOAD_MODE:-pilot}" == pilot && "${BASELOAD_VARIANT:-chp}" == chp ]]; then
    python scripts/check_baseload_chp.py \
        "results/weather-2010/${result_subdir}/networks/base_s_128___${planning_year}_b60.nc"
fi
