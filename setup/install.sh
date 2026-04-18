#!/bin/bash
set -e

# update apt and basic system deps
apt update
apt install -y git python3-pip vim

# install vLLM and other necessary libraries
pip install vllm
pip install hf_transfer

# install python dependencies
pip install -r requirements.txt

# log into huggingface for llama model
export $(cat .env)
hf auth login --token $HF_TOKEN

echo "Success!"