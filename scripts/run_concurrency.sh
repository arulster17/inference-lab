#!/bin/bash
set -e
OUTPUT_DIR=${1}
VLLM_CONFIG=${2}

if [ -z "$VLLM_CONFIG" ]; then
    echo "Usage: bash scripts/run_concurrency.sh <output_dir> <vllm-config>"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash scripts/run_concurrency.sh <output_dir> <vllm-config>"
    exit 1
fi

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

for c in 4 8 16 32 64 128 256; do
    echo "Running concurrency=$c..."
    start=$(date +%s)
    # python -m experiments.runner --concurrency $c --output $OUTPUT_DIR/c${c}.json
    PYTHONPATH=. python experiments/runner.py --concurrency $c --output $OUTPUT_DIR/c${c}.json --vllm-config $VLLM_CONFIG
    end=$(date +%s)
    echo "Concurrency=$c done in $((end - start))s"
done

echo "Done. Results in $OUTPUT_DIR"