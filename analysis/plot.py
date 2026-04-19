import json
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_FILES = {
    1:  "results/llama_c1.json",
    8:  "results/llama_c8.json",
    32: "results/llama_c32.json",
    64: "results/llama_c64.json",
}

def load_results() -> dict:
    # load each file, return dict of {concurrency: metrics}
    results = {}
    for concurrency, path in RESULT_FILES.items():
        with open(path) as f:
            result = json.load(f)
            results[concurrency] = result["metrics"]
    return results

def plot_ttft(ax, data: dict):
    # plot p50, p95, p99 lines vs concurrency on ax
    x = ["1", "8", "32", "64"]
    ax.plot(x, [data[c]["ttft_p50"] * 1000 for c in [1, 8, 32, 64]], label="p50")
    ax.plot(x, [data[c]["ttft_p95"] * 1000 for c in [1, 8, 32, 64]], label="p95")
    ax.plot(x, [data[c]["ttft_p99"] * 1000 for c in [1, 8, 32, 64]], label="p99")

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs Concurrency")
    ax.legend()
    ax.grid(True)

def plot_itl(ax, data: dict):
    # plot p50, p95, p99 lines vs concurrency on ax
    x = ["1", "8", "32", "64"]
    ax.plot(x, [data[c]["itl_p50"] * 1000 for c in [1, 8, 32, 64]], label="p50")
    ax.plot(x, [data[c]["itl_p95"] * 1000 for c in [1, 8, 32, 64]], label="p95")
    ax.plot(x, [data[c]["itl_p99"] * 1000 for c in [1, 8, 32, 64]], label="p99")

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("ITL (ms)")
    ax.set_title("ITL vs Concurrency")
    ax.legend()
    ax.grid(True)


def plot_throughput(ax, data: dict):
    # plot p50, p95, p99 lines vs concurrency on ax
    x = [1, 8, 32, 64]
    ax.plot(x, [data[c]["throughput_tps"] for c in x], marker='o')

    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs Concurrency")
    ax.grid(True)


def main():
    data = load_results()
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