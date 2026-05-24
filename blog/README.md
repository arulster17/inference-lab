# Blog Series: LLM Serving — From Fundamentals to Optimization

Each post builds on the previous. All experiments run on RunPod (single A100 80GB) with vLLM unless noted.

| # | Title | Status | Focus |
|---|-------|--------|-------|
| 1 | LLM Inference Basics | Posted | Autoregressive generation, KV cache, why memory is the binding constraint |
| 2 | PagedAttention | Draft | OS virtual memory analogy, physical vs logical blocks, eliminating fragmentation |
| 3 | Continuous Batching and the Scheduler | Posted | Naive batching problems, continuous batching, prefill vs decode, scheduler (FCFS, admission, preemption), chunked prefill as the default v1 solution |
| 4 | Baseline Setup and First Measurements | Planned | vLLM in practice, benchmark methodology, TTFT/ITL/throughput, concurrency findings |
| 5 | TBD | Planned | TBD |
| 6 | Prefix Caching | Planned | Copy-on-write from PagedAttention, shared prefix workloads, measured results |
| 7 | Quantization | Planned | FP16 vs INT8 vs INT4, throughput vs quality tradeoff, measured results |
| 8 | Speculative Decoding | Planned | Draft model setup, when it helps and when it doesn't |

---

## Notes

### vLLM version

Target audience is experienced SWEs who want to understand how modern vLLM works — they are likely running v1. The series should reflect v1 behavior rather than teaching v0 and treating optimizations as opt-in features.

Key v0 vs v1 differences that affect the series:

- **Default scheduler**: v0 prioritizes prefill — new requests get their full prompt processed before decode continues, which hurts ITL. v1 flips this: decode is prioritized and chunked prefill is on by default.
- **Chunked prefill**: opt-in in v0, on by default in v1.
- **Prefix caching**: opt-in in v0, on by default in v1.

Rather than before/after comparisons between versions, posts on chunked prefill and prefix caching should motivate the problem (why the naive approach hurts), explain the mechanism, and show why v1 made it the default. Benchmarks can be run on v1 with features selectively disabled to illustrate the cost of not having them.

Other v0 vs v1 differences:
- **Multi-step scheduling**: v1 can process multiple decode steps per scheduling iteration before returning to the scheduler, reducing overhead at high throughput. Not present in v0.
- **CUDA graphs**: v1 uses CUDA graphs more aggressively for decode steps, reducing per-step scheduling overhead. Decode ITL in v1 will be lower than v0 independently of chunked prefill — affects baseline comparisons.
- **Disaggregated prefill**: v1 has experimental support for running prefill and decode on separate instances. Out of scope for this series but worth knowing.
- **OpenAI-compatible REST API**: stable across both versions — benchmark client code does not need changes.

### Edits needed in posted blogs

**Blog 1 — scheduling section (last paragraph):** currently ends with "a good serving system needs a scheduler that can interleave prefill and decode steps to achieve predictable latency." This implies the scheduler alone solves prefill/decode interference. Should be reframed: the scheduler manages admission and ordering, but interleaving prefill and decode (chunked prefill) is what actually resolves the interference — covered in Blog 3.

**Blog 2 — "What's next" section:** currently promises Blog 3 will cover "how vLLM mixes incoming and in-progress requests." Now that Blog 3 includes chunked prefill, the teaser should reflect the fuller scope: continuous batching, the scheduler (FCFS, admission, preemption), and chunked prefill as the mechanism that resolves the prefill/decode tension.

### Benchmark methodology

- **Closed-loop vs open-loop**: the benchmark sends all requests up front and caps concurrency with a semaphore (closed-loop). Real traffic is open-loop — requests arrive independently at some rate. The two can give meaningfully different results for TTFT at high concurrency. Fine for relative comparisons but worth disclosing in Blog 4.
- **Prefix caching interference**: prefix caching is on by default in v1. When benchmarking other features, disable it explicitly (`--no-enable-prefix-caching`) to avoid shared prompt patterns in the workload skewing TTFT results.

### Quantization (Blog 7)

Largely orthogonal to the v0/v1 split. Both versions support similar backends (AWQ, GPTQ, bitsandbytes). The post should motivate the memory/compute tradeoff (FP16 vs INT8 vs INT4), explain how quantization affects weight storage and arithmetic, and benchmark throughput and quality at each level. No before/after version comparison needed — just run on v1 with different `--quantization` flags.

**Gap**: current benchmark measures latency and throughput only. Quantization also degrades output quality — need a quality metric (perplexity or task accuracy) that the current code does not support.

### Prefix Caching (Blog 6)

**Gap**: current synthetic workload uses random prompts with no shared prefixes — prefix caching will show no benefit on it. Need a new workload with a shared system prompt to demonstrate cache hits.

### Speculative Decoding (Blog 8)

Also orthogonal to v0/v1. Both versions support draft-model speculation and n-gram speculation. The post should explain when speculative decoding helps (low concurrency, predictable outputs) and when it doesn't (high concurrency, diverse outputs), with benchmarks on v1. Draft model setup is the main configuration complexity.
