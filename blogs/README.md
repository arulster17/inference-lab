# Blog Series: LLM Serving — Engineering Case Studies

Each post builds on the previous. All experiments run on Lambda Labs H100 (single GPU) with vLLM unless noted.

| # | Title | Status | Code built |
|---|-------|--------|------------|
| 1 | Measuring LLM Serving Performance from First Principles | In progress | benchmark client, workload gen, metrics |
| 2 | Continuous Batching: How vLLM Serves 10x More Requests | Planned | experiment runner, batching configs |
| 3 | Prefix Caching: Free Speedups for Repeated Prompts | Planned | shared-prefix workload, cache hit metrics |
| 4 | Quantization Tradeoffs: FP16 vs AWQ INT4 on 70B | Planned | quant configs, quality eval |
| 5 | Speculative Decoding: Using a Draft Model to Go Faster | Planned | speculative decode config, draft model setup |
| 6 | vLLM vs SGLang: Benchmark on the Same Workload | Planned | SGLang server config |
