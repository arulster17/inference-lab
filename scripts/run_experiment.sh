#!/bin/bash
set -e
OUTPUT_DIR=${1}
CONFIG=${2}

}
pkill -f "vllm serve" || true
python serving/launch.py $CONFIG
bash scripts/run_concurrency.sh results/$OUTPUT_DIR