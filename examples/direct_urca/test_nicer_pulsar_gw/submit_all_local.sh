#!/bin/bash
# Submit all SMC-RandomWalk inference jobs - Sequential Mode
# Prevent JAX from pre-allocating 90% of VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Optional: force platform allocator if still OOM
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Enable recursive globbing to scan all folders
shopt -s globstar

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Loop through all config.yaml files in any subdirectory recursively
for config_file in "$SCRIPT_DIR"/**/config.yaml; do
    [ -f "$config_file" ] || continue
    
    variant_dir="$(dirname "$config_file")"
    # Extract relative path for cleaner display logging
    relative_dir="${variant_dir#"$SCRIPT_DIR/"}"

    echo "=========================================="
    echo "Running: $relative_dir ($(date))"
    echo "=========================================="

    cd "$variant_dir" || exit 1

    run_jester_inference config.yaml

    if [ $? -eq 0 ]; then
        echo "Success: $relative_dir finished."
    else
        echo "Error: Inference failed in $relative_dir. Stopping."
        exit 1
    fi
done

echo "All done!"