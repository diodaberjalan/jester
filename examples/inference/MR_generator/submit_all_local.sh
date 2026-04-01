#!/bin/bash
# Submit all SMC-RandomWalk inference jobs - Sequential Mode

# --- Logging Setup ---
LOG_DATE=$(date +"%Y-%m-%d_%H-%M")
LOG_FILE="log_inference_${LOG_DATE}.log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "Log started: $(date)"
echo "Logging to: $LOG_FILE"

# JAX VRAM Management
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

for dir in */; do
    dir=${dir%/}

    if [ -f "$dir/submit.sh" ]; then
        echo "------------------------------------------"
        echo "Processing: $dir (Starting at $(date))"
        echo "------------------------------------------"
        
        (
            cd "$dir" || exit 1
            
            # --- Logic Cek Config File ---
            CONFIG_FILE=""
            if [ -f "config.yaml" ]; then
                CONFIG_FILE="config.yaml"
            elif [ -f "config.yml" ]; then
                CONFIG_FILE="config.yml"
            else
                echo "Skip: No config.yaml or .yml found in $dir"
                exit 0 # Lewati folder ini tapi jangan hentikan seluruh loop
            fi

            echo "Using config: $CONFIG_FILE"
            
            # Jalanin JAX inference secara sinkron
            run_jester_inference "$CONFIG_FILE"
            
            if [ $? -eq 0 ]; then
                echo "Success: $dir finished."
            else
                echo "Error: Inference failed in $dir. Stopping loop to save GPU."
                exit 1
            fi
        ) || exit 1 
    fi
done

echo "Done! All jobs finished at $(date)"