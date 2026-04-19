# Blog Series: LLM Serving — Engineering Case Studies

Each post builds on the previous. All experiments run on RunPod (single GPU) with vLLM unless noted.

> **Note:** Structure was revised after initial planning. Blog 0 was added as an accessible entry point. Original blog 1 was split into blogs 1 and 2.

| # | Title | Status | Code built |
|---|-------|--------|------------|
| 0 | Setting Up vLLM and Serving Your First LLM | Planned | install.sh, baseline.yaml, launch.py |
| 1 | Building a Benchmark from First Principles | In progress | benchmark client, workload gen, metrics |
| 2 | Diagnosing the Baseline: Prefill-Decode Interference | Planned | experiment runner, chunked prefill configs |
| 3 | Continuous Batching: How vLLM Serves 10x More Requests | Planned | batching configs |
| 4 | Prefix Caching: Free Speedups for Repeated Prompts | Planned | shared-prefix workload, cache hit metrics |
| 5 | Quantization Tradeoffs: FP16 vs AWQ INT4 on 70B | Planned | quant configs, quality eval |
| 6 | Speculative Decoding: Using a Draft Model to Go Faster | Planned | speculative decode config, draft model setup |
| 7 | vLLM vs SGLang: Benchmark on the Same Workload | Planned | SGLang server config |
