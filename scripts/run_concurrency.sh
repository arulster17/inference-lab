#!/bin/bash
set -e
OUTPUT_DIR=${1:-results/concurrency_baseline}
mkdir -p $OUTPUT_DIR

for c in 1 2 4 8 16 32 64; do
    echo "Running concurrency=$c..."
    # python -m experiments.runner --concurrency $c --output $OUTPUT_DIR/c${c}.json
    PYTHONPATH=. python experiments/runner.py --concurrency $c --output $OUTPUT_DIR/c${c}.json
done

echo "Done. Results in $OUTPUT_DIR"