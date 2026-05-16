import subprocess
import sys
import os

os.environ["PYTHONUNBUFFERED"] = "1"

def run(label, cmd):
    print(f"\n>>> {label}")
    subprocess.run(cmd, check=True)

if len(sys.argv) != 3:
    print("Usage: python scripts/run_all.py <ip> <port>")
    sys.exit(1)

ip  = sys.argv[1]
port = sys.argv[2]
key = "~/.ssh/id_ed25519"

ssh = ["ssh", "t", f"root@{ip}", "-p", port, "-i", key]
scp = ["scp", "-P", port, "-i", key]

print(f"Starting full experiment pipeline on {ip}:{port}")

run("Clearing old files on pod...",   ssh + ["rm -rf inference-lab"])
run("Cloning repo on pod...",         ssh + ["git clone --branch 04-baseline https://github.com/arulster17/inference-lab.git"])
run("Copying .env to pod...",         scp + [".env", f"root@{ip}:~/inference-lab/"])
run("Installing dependencies...",     ssh + ["cd inference-lab && bash setup/install.sh"])
run("Running experiment...",          ssh + ["cd inference-lab && bash scripts/run_experiment.sh"])
run("Copying results locally...",     scp + ["-r", f"root@{ip}:~/inference-lab/results/concurrency_baseline/", "results/concurrency_baseline/"])
run("Plotting...",                    ["python", "analysis/plot.py", "--results", "results/concurrency_baseline"])

print("\nDone!")
