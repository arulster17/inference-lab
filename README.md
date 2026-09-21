# inference-lab

A hands-on lab for understanding LLM serving: benchmark vLLM under different
configurations, measure the effect of features like chunked prefill, and write
up what actually changes and why.

Companion to a blog series on LLM serving, from first principles up through
production optimizations. All experiments run on a single A100 80GB via
RunPod, serving Llama 3.1 8B with vLLM.

## What's here

- **`serving/`** — YAML-driven launcher (`launch.py`) that starts a vLLM server
  from a config, waits for health, and sends a warmup request. Configs pair a
  baseline against one changed knob (`baseline.yaml` vs
  `no_chunked_prefill.yaml`).
- **`benchmark/`** — async load-testing client. Streams chat completions
  concurrently (bounded by a semaphore), recording per-request TTFT
  (time-to-first-token) and ITL (inter-token latency).
- **`experiments/`** — `runner.py`, one benchmark point: builds the workload,
  drives the client at a given concurrency, writes metrics JSON plus raw
  per-request timings as `.npz`.
- **`scripts/`** — pod automation. `run_all.py` is the full pipeline: clone on a
  RunPod box, install, sweep concurrency 4→256, copy results back, plot.
- **`analysis/`** — `plot.py`: TTFT/ITL percentiles and throughput vs.
  concurrency, plus ITL KDE comparisons between two runs.
- **`results/`** — recorded sweeps (`baseline`, `chunked_2048`,
  `no_chunked_2048`, …) used in the blog posts.
- **`blog/`** — the write-ups. See below.

## Blog series: LLM Serving — From Fundamentals to Optimization

Each post builds on the last. Full status and notes in [`blog/README.md`](blog/README.md).

| # | Title | Status |
|---|-------|--------|
| 1 | [LLM Inference Basics](blog/1-basics/01-inference-basics.md) — autoregressive generation, the KV cache, why memory is the binding constraint | [Posted](https://medium.com/@arulster17/inference-basics-and-why-model-serving-is-hard-51814cb6b069) |
| 2 | [PagedAttention](blog/2-paged-attention/02-paged-attention.md) — the OS virtual memory analogy, physical vs. logical blocks, eliminating fragmentation | Draft |
| 3 | [Continuous Batching and the Scheduler](blog/3-batching-and-scheduler/03-continuous-batching.md) — naive batching, continuous batching, prefill vs. decode, chunked prefill | Posted |
| 4 | [Baseline Setup and First Measurements](blog/4-baseline/04-baseline.md) — vLLM in practice, benchmark methodology, TTFT/ITL/throughput | Draft |
| 5 | Speculative Decoding — spending idle low-concurrency compute to cut latency | Planned |
| 6 | Prefix Caching — copy-on-write from PagedAttention, shared-prefix workloads | Planned |
| 7 | Quantization — FP16 vs. INT8 vs. INT4, throughput vs. quality | Planned |

## Running it

Needs an NVIDIA GPU; everything here was run on a single A100 80GB RunPod box.
A `.env` with `HF_TOKEN` is required to pull the Llama weights.

**Whole pipeline against a pod** — clones, installs, sweeps, copies results back,
plots:

```bash
python scripts/run_all.py <pod-ip> <ssh-port> baseline serving/configs/baseline.yaml
```

**Or step by step, on the box itself:**

```bash
bash setup/install.sh

# start vLLM from a config, wait for health, warm it up
python serving/launch.py serving/configs/baseline.yaml

# sweep concurrency 4 -> 256 into results/baseline/
bash scripts/run_concurrency.sh results/baseline serving/configs/baseline.yaml

# or a single point
PYTHONPATH=. python experiments/runner.py \
  --concurrency 32 \
  --output results/baseline/c32.json \
  --vllm-config serving/configs/baseline.yaml
```

**Plot a sweep** (figures land in `analysis/<run-name>/`):

```bash
python analysis/plot.py --results results/baseline
python analysis/plot.py --results results/chunked_2048 --histogram \
  --compare results/no_chunked_2048 --label1 chunked --label2 "no chunked"
```
