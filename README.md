# inference-lab

A hands-on lab for understanding LLM serving: benchmark vLLM under different
configurations, measure the effect of features like chunked prefill, and write
up what actually changes and why.

Companion to a blog series on LLM serving, from first principles up through
production optimizations. All experiments run on a single A100 80GB via
RunPod, serving Llama 3.1 8B with vLLM.

## What's here

- **`serving/`** — YAML-driven launcher (`launch.py`) that starts a vLLM
  server from a config file, waits for health, and sends a warmup request.
- **`benchmark/`** — async load-testing client. Streams chat completions
  concurrently (bounded by a semaphore), and records per-request TTFT
  (time-to-first-token) and ITL (inter-token latency).
- **`analysis/`** — turns raw benchmark JSON into plots (p50/p95/p99 latency,
  throughput vs. concurrency).
- **`experiments/`** — sweep runner that drives the benchmark client across
  configurations (e.g. concurrency 1 → 64, chunked vs. unchunked prefill).
- **`results/`** — recorded runs (`results_clean`, `results-chunked`,
  `results-unchunked`) used in the blog posts.
- **`blog/`** — the write-ups. See below.

## Blog series: LLM Serving — From Fundamentals to Optimization

Each post builds on the last.

| # | Title | Status |
|---|-------|--------|
| 1 | [LLM Inference Basics](blog/1-basics/01-inference-basics.md) — autoregressive generation, the KV cache, why memory is the binding constraint | [Posted](https://medium.com/@arulster17/inference-basics-and-why-model-serving-is-hard-51814cb6b069) |
| 2 | [PagedAttention](blog/2-paged-attention/02-paged-attention.md) — the OS virtual memory analogy, physical vs. logical blocks, eliminating fragmentation | Draft |
| 3 | [Continuous Batching and the Scheduler](blog/3-batching-and-scheduler/03-continuous-batching.md) — prefill vs. decode, how requests move through the system | Draft |
| 4 | Baseline Setup and First Measurements — vLLM in practice, benchmark methodology, TTFT/ITL/throughput | Planned |
| 5 | Chunked Prefill — what it changes, measured results | Planned |
| 6 | Prefix Caching — copy-on-write from PagedAttention, shared-prefix workloads | Planned |
| 7 | Quantization — FP16 vs. INT8 vs. INT4, throughput vs. quality | Planned |
| 8 | Speculative Decoding — when it helps, when it doesn't | Planned |

## Running it

```bash
pip install -e .

# start a vLLM server from a config
python serving/launch.py serving/configs/baseline.yaml

# run a single benchmark point against it
python experiments/runner.py --concurrency 32 --output results/llama_c32.json

# plot TTFT/ITL/throughput vs. concurrency (expects results/llama_c{1,8,32,64}.json)
python analysis/plot.py
```

Requires an NVIDIA GPU with vLLM installed; benchmarks were run on RunPod.
