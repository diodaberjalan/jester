#!/bin/bash

# Generate a timestamp (Format: YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="submission_log_${TIMESTAMP}.txt"

{
    echo "--- Starting submissions at $(date) ---"
    
    for dir in */; do
        dir=${dir%/}  # Remove trailing slash

        # Submit if submit.sh exists
        if [ -f "$dir/submit.sh" ]; then
            echo "Submitting $dir"
            cd "$dir"
            sbatch submit.sh
            cd ..
        fi
    done
    
    echo "--- Finished submissions ---"
} | tee "$LOG_FILE"