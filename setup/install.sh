#!/bin/bash
set -e

# update apt and basic system deps
sudo apt update
sudo apt install -y git python3-pip

# install vLLM and other necessary libraries
pip install vllm
pip install hf_transfer

# install python dependencies
pip install -r requirements.txt

# log into huggingface for llama model
hf auth login --token $HF_TOKEN

echo "Success!"