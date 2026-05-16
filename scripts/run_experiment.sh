#!/bin/bash
set -e
pkill -f "vllm serve" || true
python serving/launch.py serving/configs/baseline.yaml
bash scripts/run_concurrency.sh results/concurrency_baseline