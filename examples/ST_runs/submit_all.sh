#!/bin/bash

# Generate a timestamp (Format: YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="submission_log_${TIMESTAMP}.txt"

{
    echo "--- Starting submissions at $(date) ---"

    P_DIR=$(basename "$PWD")

    # Find all submit.sh nested under parent date folders (parent/output/submit.sh)
    for submit_path in */*/submit.sh; do
        dir=$(dirname "$submit_path")

        echo "Submitting $dir"
        cd "$dir"

        # Make sure outdir exists so SLURM doesn't complain
        mkdir -p outdir
        sbatch --job-name="${P_DIR}/${dir}" --output="./outdir/log_${P_DIR}_$(basename ${dir}).out" submit.sh

        cd - > /dev/null
    done

    echo "--- Finished submissions ---"
} | tee "$LOG_FILE"
