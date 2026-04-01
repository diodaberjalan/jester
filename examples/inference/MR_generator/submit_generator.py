import os

def generate_submit_script(folder_name):
    """Generates a SLURM submit.sh script inside the specified folder."""
    
    submit_script_content = f"""#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition gpu
#SBATCH -t 1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=40G
#SBATCH --output="./outdir/log_smc_rw.out"
#SBATCH --job-name="{folder_name}"

now=$(date)
echo "$now"

module load arch/r1/x86_64
module load python/3.11.7
module load miniforge3/4.8.3-4
module load cuda/11.8.0
module load texlive/20240312
source activate jester-MG
# Display GPU name
nvidia-smi --query-gpu=name --format=csv,noheader

echo "=========================================="
echo "=== Running jester inference ({folder_name}) ==="
echo "=========================================="

uv run run_jester_inference config.yml
"""
    submit_path = os.path.join(folder_name, "submit.sh")
    with open(submit_path, "w") as f:
        f.write(submit_script_content)