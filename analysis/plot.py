import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from matplotlib.ticker import ScalarFormatter
import numpy as np
  


def load_results(results_dir: str) -> dict:
    # load each file, return dict of {concurrency: metrics}
    results = {}
    for path in sorted(Path(results_dir).glob("c[0-9]*.json")):                                                                      
        with open(path) as f:
            metrics = json.load(f)
        c = int(path.stem[1:])
        results[c] = metrics
    return results

def load_raw(results_dir: str) -> dict:
    raw = {}
    for path in sorted(Path(results_dir).glob("c[0-9]*.npz")):
        c = int(path.stem[1:])
        raw[c] = np.load(path)
    return raw

def plot_ttft(ax, data: dict):
    # plot p50, p95, p99 lines vs concurrency on ax
    x = sorted(data.keys())
    ax.plot(x, [data[c]["ttft_p50"] * 1000 for c in x], label="p50")
    ax.plot(x, [data[c]["ttft_p95"] * 1000 for c in x], label="p95")
    ax.plot(x, [data[c]["ttft_p99"] * 1000 for c in x], label="p99")

    ax.set_xlabel("Concurrency")
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
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
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_ylabel("ITL (ms)")
    ax.set_title("ITL vs Concurrency")
    ax.legend()
    ax.grid(True)


def plot_throughput(ax, data: dict):
    # plot p50, p95, p99 lines vs concurrency on ax
    x = sorted(data.keys())
    ax.plot(x, [data[c]["throughput_tps"] for c in x], marker='o')

    ax.set_xlabel("Concurrency")
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs Concurrency")
    ax.grid(True)


def plot_itl_histograms(raw: dict):
    concurrencies = sorted(raw.keys())
    n = len(concurrencies)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, c in zip(axes, concurrencies):
        ax.hist(raw[c]["itl"] * 1000, bins=100)
        ax.set_title(f"c={c}")
        ax.set_xlabel("ITL (ms)")
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--histogram", action="store_true")
    args = parser.parse_args()
    data = load_results(args.results)

    prefix =  Path("analysis") / Path(args.results).name
    prefix.mkdir(parents=True, exist_ok=True)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_ttft(ax1, data)
    plot_itl(ax2, data)
    plt.tight_layout()
    plt.savefig(prefix / "ttft_itl.png")

    fig2, ax3 = plt.subplots(figsize=(6, 5))
    plot_throughput(ax3, data)
    plt.savefig(prefix / "throughput.png")

    if args.histogram:
        raw = load_raw(args.results)
        fig = plot_itl_histograms(raw)
        fig.savefig(prefix / "itl_histogram.png")

if __name__ == "__main__":
    main()