import os
import sys
import time
import subprocess
import runpod
import argparse
from dotenv import load_dotenv

load_dotenv()

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


parser = argparse.ArgumentParser()
parser.add_argument("output_folder", type=str)
parser.add_argument("config", type=str)
parser.add_argument("--volume-id", type=str, default=None)

args = parser.parse_args()
output_folder = args.output_folder
config = args.config
runpod.api_key = os.environ["RUNPOD_API_KEY"]

pod = runpod.create_pod(name=f"inference-lab-{output_folder}", 
                        image_name="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04", 
                        gpu_type_id="NVIDIA A100-SXM4-80GB", 
                        gpu_count=1, 
                        container_disk_in_gb=50, 
                        ports="22/tcp",
                        network_volume_id=args.volume_id,)

pod_id = pod["id"]
print(f"Pod created: {pod_id}")

try:
    ip, port = wait_for_running(pod_id)
    subprocess.run(["python", "scripts/run_all.py", ip, str(port), output_folder, config], check=True)
finally:
    print("Terminating pod...")
    runpod.terminate_pod(pod_id)
