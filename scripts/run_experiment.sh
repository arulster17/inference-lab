#!/bin/bash
set -e
python serving/launch.py serving/configs/baseline.yaml
bash scripts/run_concurrency.sh results/concurrency_baseline