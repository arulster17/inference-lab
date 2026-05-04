#!/bin/bash
set -e

IP=$1
PORT=$2
KEY=~/.ssh/id_ed25519

if [ -z "$IP" ] || [ -z "$PORT" ]; then
    echo "Usage: bash scripts/setup_pod.sh <ip> <port>"
    exit 1
fi

echo "Cloning repo..."
ssh root@$IP -p $PORT -i $KEY "git clone https://github.com/arulster17/inference-lab.git && cd inference-lab && git switch blog/01-baseline"

echo "Copying .env..."
scp -P $PORT -i $KEY C:/Users/Arul/Projects/inference-lab/.env root@$IP:~/inference-lab/

echo "Running install script..."
ssh root@$IP -p $PORT -i $KEY "cd inference-lab && bash setup/install.sh"

echo "Done! Pod is ready."
