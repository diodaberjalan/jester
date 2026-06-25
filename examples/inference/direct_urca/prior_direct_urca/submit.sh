#!/bin/bash
#SBATCH --job-name="direct-urca-trigger"
#SBATCH --output="direct_urca_%j.out"
#SBATCH --error="direct_urca_%j.err"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

run_jester_inference config.yaml
