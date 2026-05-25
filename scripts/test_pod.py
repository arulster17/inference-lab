"""
Quick test: create a RunPod pod, create a hello-world file via SSH, wait for input, terminate it.

Usage:
    python scripts/test_pod.py

Requires RUNPOD_API_KEY in .env
"""

import os
import sys
import time
import subprocess
import runpod
from dotenv import load_dotenv

load_dotenv()

SSH_KEY        = "~/.ssh/id_ed25519"
POD_NAME       = "inference-lab-test"
IMAGE          = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
GPU_TYPE       = "NVIDIA A100-SXM4-80GB"
POLL_INTERVAL  = 5  # seconds between status checks


def get_ssh_info(pod: dict) -> tuple[str, int] | None:
    ports = (pod.get("runtime") or {}).get("ports") or []
    for p in ports:
        if p.get("privatePort") == 22:
            return p["ip"], p["publicPort"]
    return None


def ssh(ip: str, port: int, cmd: str, retries: int = 10):
    for attempt in range(retries):
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             "-i", SSH_KEY, "-p", str(port), f"root@{ip}", cmd],
        )
        if result.returncode == 0:
            return
        print(f"SSH not ready, retrying ({attempt + 1}/{retries})...")
        time.sleep(5)
    raise RuntimeError("SSH failed after all retries")


def wait_for_running(pod_id: str) -> tuple[str, int]:
    print("Waiting for pod to be running", end="", flush=True)
    while True:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus")
        ssh_info = get_ssh_info(pod)
        if status == "RUNNING" and ssh_info:
            print(" ready.")
            return ssh_info
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


def main():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY not set in .env")
        sys.exit(1)

    runpod.api_key = api_key

    print(f"Creating pod ({GPU_TYPE})...")
    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=IMAGE,
        gpu_type_id=GPU_TYPE,
        gpu_count=1,
        container_disk_in_gb=50,
        ports="22/tcp",
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    try:
        ip, port = wait_for_running(pod_id)
        print(f"SSH: {ip}:{port}")

        print("Creating hello-world file on pod...")
        ssh(ip, port, "echo 'hello world' > /tmp/test.txt && cat /tmp/test.txt")

        input("\nPress Enter to terminate the pod...")
    finally:
        print("Terminating pod...")
        runpod.terminate_pod(pod_id)
        print("Done.")


if __name__ == "__main__":
    main()
