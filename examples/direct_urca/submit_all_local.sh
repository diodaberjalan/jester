#!/bin/bash
# Submit all SMC-RandomWalk inference jobs - Sequential Mode
# Prevent JAX from pre-allocating 90% of VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Optional: force platform allocator if still OOM
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for model in skyrme metamodel; do
    for variant_dir in "$SCRIPT_DIR/$model"/*/; do
        [ -d "$variant_dir" ] || continue
        config_file="$variant_dir/config.yaml"
        [ -f "$config_file" ] || continue

        variant_name="$(basename "$variant_dir")"
        echo "=========================================="
        echo "Running: $model/$variant_name ($(date))"
        echo "=========================================="

        cd "$variant_dir" || exit 1

        run_jester_inference config.yaml

        if [ $? -eq 0 ]; then
            echo "Success: $model/$variant_name finished."
        else
            echo "Error: Inference failed in $model/$variant_name. Stopping."
            exit 1
        fi
    done
done

echo "All done!"