#!/usr/bin/env bash
#SBATCH --job-name=baseload-sweep
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=baseload-controller-%j.out

# Submit this file from the pypsa-eur repository root.
# The small allocation runs Snakemake; Snakemake submits compute jobs to SLURM.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pypsa-eur repository root}"

if [[ ! -f Snakefile || ! -f config/config.baseload-weather.yaml ]]; then
    echo "Submit from the pypsa-eur repository root." >&2
    exit 1
fi

# If Conda is provided by a cluster module, load that module here first.
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

# Use the sibling licence file when present; otherwise keep the cluster's setup.
if [[ -z "${GRB_LICENSE_FILE:-}" && -f ../gurobi/gurobi.lic ]]; then
    export GRB_LICENSE_FILE="$PWD/../gurobi/gurobi.lic"
fi
python -c 'import snakemake_executor_plugin_slurm, gurobipy'

if [[ ! -f data/WDPA.gpkg || ! -f data/WDPA_WDOECM_marine.gpkg ]]; then
    echo "Warning: WDPA inputs are absent; this PyPSA-Eur version may fail to retrieve them." >&2
fi

case "${BASELOAD_MODE:-pilot}" in
    pilot)
        target=results/weather-2013/baseload/metrics/base_s_128___2045_b0.csv
        ;;
    full)
        target=baseload_sweep
        ;;
    *)
        echo "BASELOAD_MODE must be pilot or full." >&2
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

echo "Mode: ${BASELOAD_MODE:-pilot}; target: $target"
echo "Repository: $PWD; environment: $CONDA_DEFAULT_ENV"

snakemake --snakefile Snakefile \
    --configfile config/config.baseload-weather.yaml \
    --software-deployment-method conda \
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
    "$target"
