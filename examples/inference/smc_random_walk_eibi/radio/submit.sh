#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu
#SBATCH -t 10:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=40G
#SBATCH --output="./outdir/log_smc_rw.out"
#SBATCH --job-name="SMC-RW-radio"

now=$(date)
echo "$now"

module load  arch/r1/x86_64
module load python/3.11.7
module load miniforge3/4.8.3-4
module load cuda/11.8.0
module load texlive/20240312
source activate jester-MG
# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=========================================="
echo "=== Running jester inference (SMC-RW) ==="
echo "=========================================="

run_jester_inference config.yaml