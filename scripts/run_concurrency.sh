#!/bin/bash
set -e
OUTPUT_DIR=${1}
if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash scripts/run_concurrency.sh <output_dir>"
    exit 1
fi

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

for c in 128 256; do
    echo "Running concurrency=$c..."
    start=$(date +%s)
    # python -m experiments.runner --concurrency $c --output $OUTPUT_DIR/c${c}.json
    PYTHONPATH=. python experiments/runner.py --concurrency $c --output $OUTPUT_DIR/c${c}.json
    end=$(date +%s)
    echo "Concurrency=$c done in $((end - start))s"
done

echo "Done. Results in $OUTPUT_DIR"