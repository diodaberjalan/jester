#!/bin/bash

# Generate a timestamp (Format: YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="submission_log_${TIMESTAMP}.txt"

{
    echo "--- Starting submissions at $(date) ---"

    P_DIR=$(basename "$PWD")

    for dir in */; do
        dir=${dir%/}  # Remove trailing slash

        # Submit if submit.sh exists
        if [ -f "$dir/submit.sh" ]; then
            echo "Submitting $dir"
            cd "$dir"

            # Make sure outdir exists so SLURM doesn't complain
            mkdir -p outdir
            sbatch --job-name="${P_DIR}/${dir}" --output="./outdir/log_${P_DIR}_${dir}.out" submit.sh

            cd ..
        fi
    done

    echo "--- Finished submissions ---"
} | tee "$LOG_FILE"
