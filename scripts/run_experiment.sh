#!/bin/bash
set -e
OUTPUT_DIR=${1}
pkill -f "vllm serve" || true
python serving/launch.py serving/configs/baseline.yaml
bash scripts/run_concurrency.sh results/$OUTPUT_DIR