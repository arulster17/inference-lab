# inference-lab

Benchmark harness and experiments behind the Medium series
[Understanding LLM Serving](https://medium.com/understanding-llm-serving).

It launches vLLM on a RunPod GPU, sends a synthetic workload at fixed concurrency levels,
records per-request TTFT and every inter-token latency (ITL), and plots the results.

## Headline result

Llama 3.1 8B Instruct (FP16) on one A100 80GB SXM, vLLM 0.19.0, prefix caching off,
concurrency up to 256:

- **Without chunked prefill**, ITL is bimodal. Most decode steps take about 25 ms, but any
  step that carries a 2048-token prefill stalls every in-flight request to about 200 ms.
- **With chunked prefill** (512-token budget), the spikes are gone and ITL settles around a
  steady 65 ms, at the cost of a higher typical ITL and higher TTFT.

Full write-up: post 4, "Benchmarking vLLM". Plots are in `analysis/`.

## Layout

| Path | What it holds |
|---|---|
| `benchmark/` | Async client, workload generator, latency metrics |
| `serving/` | vLLM launcher and the server configs used for each run |
| `experiments/` | Concurrency sweep runner |
| `scripts/` | Pod lifecycle and the end-to-end pipeline (`run_all.py`) |
| `setup/` | Pod install script |
| `results/` | Raw results per run |
| `analysis/` | Plotting script and generated charts |
| `blog/` | Post drafts and series notes |

## Running

```
python scripts/run_all.py <pod-ip> <ssh-port> <output-folder> <vllm-config>
```

This clones the repo on the pod, installs dependencies, runs the sweep, copies results back
and plots them. It expects a RunPod pod reachable over SSH and a `.env` file with the
required tokens.
