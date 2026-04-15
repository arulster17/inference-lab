#!/bin/bash
set -e

# update apt and basic system deps
sudo apt update
sudo apt install -y git python3-pip

# install vLLM
pip install vllm

# install python dependencies
pip install -r requirements.txt

# log into huggingface for llama model
huggingface-cli login

echo "Success!"