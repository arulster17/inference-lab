# Blog 1: Measuring LLM Serving Performance from First Principles

**Status:** In progress
**Thesis:** Most people use black-box benchmark tools without understanding what's being measured or why. This post builds the measurement layer from scratch.

---

## What to cover

### 1. Why these metrics matter
- TTFT: captures prefill cost, drives perceived responsiveness
- ITL: captures decode speed, driven by memory bandwidth
- Throughput (tokens/s, req/s): capacity of the system under load
- Why total latency alone is not enough

### 2. Why SSE is required
- Blocking HTTP = one timestamp, can't decompose TTFT vs ITL
- SSE = per-token timestamps, full latency breakdown visible
- Show the raw wire format (data: {...} chunks)

### 3. What we built
- Async benchmark client: N concurrent streaming requests, timestamps each SSE chunk
- Workload generator: synthetic (fixed lengths) and ShareGPT (realistic)
- Metrics aggregator: p50/p95/p99 for TTFT and ITL, req/s and tok/s

### 4. Baseline setup
- Model: Llama-3.1-8B-Instruct, FP16
- Server: vLLM, default settings, single H100
- Nothing tuned — this is the "before"

### 5. Results
- Chart: TTFT p50/p95/p99 at 1, 8, 32, 64 concurrent requests
- Chart: ITL p50/p95/p99 at same concurrency levels
- Observations: where does latency blow up? What's the bottleneck?

### 6. What this sets up
- These numbers are the baseline every future post compares against
- Tease: "what happens when we turn on continuous batching?"

---

## What we build (code)
- `benchmark/client.py` — async SSE client ✓
- `benchmark/workload.py` — workload generator ✓
- `benchmark/metrics.py` — aggregation and stats ✓
- `serving/configs/baseline.yaml` — vLLM server config
- `serving/launch.py` — reads config YAML, starts vLLM with right args
- `experiments/runner.py` — wires workload + client + metrics, saves results
- `setup/install.sh` — Lambda Labs instance setup (CUDA, vLLM, HuggingFace)
- `analysis/plot.py` — TTFT and ITL charts at 1/8/32/64 concurrency
