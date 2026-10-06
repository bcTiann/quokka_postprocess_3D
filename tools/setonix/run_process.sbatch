#!/bin/bash
#SBATCH --job-name=quokka2s-process
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%x-%j.log
#SBATCH --export=ALL

# Submit from the repository root after activating its Python environment.
# Create output/slurm before sbatch opens the log file.
# Supply your account and CPU partition in the sbatch command.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
config_file="${1:-configs/emission_process.yaml}"
python_executable="$(command -v python)"

# Parallelism comes from the YAML's query and spectral workers.
# Keep each NumPy/SciPy native library call from starting another thread pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Repository: ${SLURM_SUBMIT_DIR}"
echo "Commit: $(git rev-parse HEAD)"
echo "Python: ${python_executable}"
"${python_executable}" --version
echo "Configuration: ${config_file}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"

# One Python process reads slabs and uses the threads selected in the YAML.
exec srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --distribution=block:block:block \
    --cpu-bind=cores \
    "${python_executable}" -u -m quokka2s.process_snapshot \
    --config "${config_file}"
