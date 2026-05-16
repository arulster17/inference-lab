import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_results(results_dir: str) -> dict:
    # load each file, return dict of {concurrency: metrics}
    results = {}
    for path in sorted(Path(results_dir).glob("*.json")):                                                                      
        with open(path) as f:
            result = json.load(f)
        c = result["config"]["concurrency"]
        results[c] = result["metrics"]
    return results

def plot_ttft(ax, data: dict):
    # plot p50, p95, p99 lines vs concurrency on ax
    x = sorted(data.keys())
    ax.plot(x, [data[c]["ttft_p50"] * 1000 for c in x], label="p50")
    ax.plot(x, [data[c]["ttft_p95"] * 1000 for c in x], label="p95")
    ax.plot(x, [data[c]["ttft_p99"] * 1000 for c in x], label="p99")

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs Concurrency")
    ax.legend()
    ax.grid(True)

def plot_itl(ax, data: dict):
    # plot p50, p95, p99 lines vs concurrency on ax
    x = sorted(data.keys())
    ax.plot(x, [data[c]["itl_p50"] * 1000 for c in x], label="p50")
    ax.plot(x, [data[c]["itl_p95"] * 1000 for c in x], label="p95")
    ax.plot(x, [data[c]["itl_p99"] * 1000 for c in x], label="p99")

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("ITL (ms)")
    ax.set_title("ITL vs Concurrency")
    ax.legend()
    ax.grid(True)


def plot_throughput(ax, data: dict):
    # plot p50, p95, p99 lines vs concurrency on ax
    x = sorted(data.keys())
    ax.plot(x, [data[c]["throughput_tps"] for c in x], marker='o')

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs Concurrency")
    ax.grid(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    data = load_results(args.results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_ttft(ax1, data)
    plot_itl(ax2, data)
    plt.tight_layout()
    plt.savefig("analysis/ttft_itl.png")

    fig2, ax3 = plt.subplots(figsize=(6, 5))
    plot_throughput(ax3, data)
    plt.savefig("analysis/throughput.png")

if __name__ == "__main__":
    main()