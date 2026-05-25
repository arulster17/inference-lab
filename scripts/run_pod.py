import os
import sys
import time
import subprocess
import runpod
from dotenv import load_dotenv

load_dotenv()

SSH_KEY = "~/.ssh/id_ed25519"
POD_NAME = "inference-lab"
IMAGE = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
GPU_TYPE = "NVIDIA A100-SXM4-80GB"


def get_ssh_info(pod):
    ports = (pod.get("runtime") or {}).get("ports") or []
    for p in ports:
        if p["privatePort"] == 22:
            return p["ip"], p["publicPort"]
    return None


def wait_for_running(pod_id):
    print("Waiting for pod", end="", flush=True)
    while True:
        pod = runpod.get_pod(pod_id)
        info = get_ssh_info(pod)
        if pod["desiredStatus"] == "RUNNING" and info:
            print(" ready.")
            return info
        print(".", end="", flush=True)
        time.sleep(3)


if len(sys.argv) != 3:
    print("Usage: python scripts/run_pod.py <output_folder> <vllm-config>")
    sys.exit(1)

runpod.api_key = os.environ["RUNPOD_API_KEY"]
output_folder, config = sys.argv[1], sys.argv[2]

pod = runpod.create_pod(name=POD_NAME, 
                        image_name=IMAGE, 
                        gpu_type_id=GPU_TYPE, 
                        gpu_count=1, 
                        container_disk_in_gb=50, 
                        ports="22/tcp")

pod_id = pod["id"]
print(f"Pod created: {pod_id}")

try:
    ip, port = wait_for_running(pod_id)
    subprocess.run(["python", "scripts/run_all.py", ip, str(port), output_folder, config], check=True)
finally:
    print("Terminating pod...")
    runpod.terminate_pod(pod_id)
