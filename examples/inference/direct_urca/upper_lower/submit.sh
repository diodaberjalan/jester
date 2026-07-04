#!/bin/bash
#SBATCH --job-name="direct-urca-upper-lower"
#SBATCH --output="direct_urca_upper_lower_%j.out"
#SBATCH --error="direct_urca_upper_lower_%j.err"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

run_jester_inference config.yaml
