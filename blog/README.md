# Blog Series: LLM Serving — From Fundamentals to Optimization

Each post builds on the previous. All experiments run on RunPod (single A100 80GB) with vLLM unless noted.

| # | Title | Status | Focus |
|---|-------|--------|-------|
| 1 | LLM Inference Basics | Posted | Autoregressive generation, KV cache, why memory is the binding constraint |
| 2 | PagedAttention | Draft | OS virtual memory analogy, physical vs logical blocks, eliminating fragmentation |
| 3 | Continuous Batching and the Scheduler | Draft | Prefill vs decode, how requests move through the system, latency/throughput tradeoff |
| 4 | Baseline Setup and First Measurements | Planned | vLLM in practice, benchmark methodology, TTFT/ITL/throughput, concurrency findings |
| 5 | Chunked Prefill | Planned | What it changes and why, measured results |
| 6 | Prefix Caching | Planned | Copy-on-write from PagedAttention, shared prefix workloads, measured results |
| 7 | Quantization | Planned | FP16 vs INT8 vs INT4, throughput vs quality tradeoff, measured results |
| 8 | Speculative Decoding | Planned | Draft model setup, when it helps and when it doesn't |
