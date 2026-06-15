# Baseline

In the previous posts, we discussed the theory of how vLLM manages requests efficiently. However, without measurements, there's no way to know how much an optimization actually helps. Before we can discuss further optimizations for LLM serving, we need a benchmark harness that can quantify their impacts. In this post, we build this harness, take baseline measurements, and test chunked prefill, an optimization from the previous post.
---


## Setup

All experiments are run on a single A100 80GB SXM GPU on RunPod and use the Llama 3.1 8B Instruct model with vLLM 0.19.0. For our baseline, we run the model in FP16 with prefix caching and chunked prefill disabled. More details and the full code are in the inference-lab repo (link).

## Benchmark methodology

We use a *closed-loop* benchmark, meaning we have a fixed pool of requests with a cap on how many can be active at once. In the real world, traffic arrives at some rate regardless of what the server is doing. Closed-loop ignores the arrival process and directly sets the server load, making it the best option for measuring the server itself. For the baseline, each request has a 512-token prompt and a 256-token maximum output. Fixing both lengths means we only change the concurrency across experiments.

For now, we will track three metrics for each run:

- **TTFT (time to first token)**: how long it takes for a request to receive its first output token.
- **ITL (inter-token latency)**: time between consecutive output tokens.
- **Throughput**: total output tokens per second across all requests.

For each latency metric, we look at the 50th, 95th, and 99th percentiles rather than just the mean, which is skewed by outliers. The 50th percentile tells us what a typical request experiences, and the 95th and 99th percentiles tell us what the worst requests experience. We run 4096 requests per concurrency level to keep tail percentiles stable.


# Results

## Throughput

throughput image /analysis/baseline/throughput.png

Throughput rises steeply until around c=128, where it flattens. At c=1, the server processes one request at a time and generates around 90 tokens per second. A single request's decode steps do not produce enough work to keep the thousands of GPU cores busy, so most of the hardware sits idle. As concurrency increases, the scheduler batches more requests into each forward pass, and GPU utilization climbs. By c=128, throughput has saturated at around 1880 tokens per second, as each forward pass is fully using the GPU's compute and memory bandwidth, meaning adding more requests to the batch will not improve performance. 

This is the main payoff of serving requests concurrently: the same GPU serving one request at a time is ~20x less efficient than when serving 128 concurrent requests.


## Latency

ttft itl image /analysis/baseline/ttft_itl.png

Unfortunately, the added throughput comes at a price.

**TTFT** grows steeply with concurrency as each new request must join a queue behind all active requests, and each step in the queue takes longer as the batch grows. It stays nearly flat until c=64, then explodes. Between c=64 and c=128, it jumps to 2.5 seconds, and at c=256, the p50, p95, and p99 lines have converged above 13 seconds, meaning a majority of requests are stuck in the prefill queue instead of generating tokens.

The shape of the graph comes from a phase change at saturation:

- **Below saturation (c <= ~64)**: A new request joins the next forward pas immediately. TTFT is roughly the duration of one forward pass. Adding sequences to the batch is nearly free until the GPU is actually saturated, so TTFT is dominated by the overhead.

- **Above saturation (c >= ~128)**: Throughput has plateaued, meaning the GPU physically cannot serve all active requests in the same step.The scheduler admits new requests only as fast as old ones finish, and the rest sit in vLLM's internal queue. TTFT is now dominated by the queue wait time, which scales with the number of in-flight requests.

The **ITL** also grows, as each decode step is shared across all active requests, meaning each step takes proportionally longer as concurrency rises. At high concurrency, it can reach around ~40 ms, meaning a 256-token response will take 10 seconds to stream out after the first token arrives. At these concurrencies, the decode phase itself becomes a significant source of latency.



## Measuring Chunked Prefill

The previous post explained the basics of chunked prefill: instead of processing a large prompt in a monolithic prefill step, vLLM splits it into fixed-size chunks and interleaves them with decode work. The prediction was that this would smooth out ITL, since large prefills would no longer take priority over active decode steps and spike the latencies of all active requests.

We run the sweep with 2048-token prompts and a 









The last post explained how chunked prefill works: instead of processing a large prompt in one monolithic prefill step, vLLM splits it into fixed-size chunks and interleaves them with decode work. The prediction was that this would smooth out ITL, since large prefills would no longer monopolize a decode step and spike the latency of every in-flight request.

We rerun the sweep with 2048-token prompts and compare chunked prefill on versus off. We use longer prompts because the effect only becomes visible when the prefill step is large relative to a decode step. At 512 tokens, a prefill completes quickly enough that it barely disturbs the decode cadence. At 2048 tokens, a single unchunked prefill dominates an entire forward pass and the disruption shows clearly in the latency distribution. We look at c=64, which sits at the knee of the throughput curve and represents a realistic operating point for a loaded server.

![ITL Distribution at c=64](../../analysis/chunked_2048/itl_histogram_64.png)

Without chunked prefill, the distribution is *bimodal*: most decode steps are fast pure-decode steps clustered around 25ms, but when a 2048-token prefill arrives it monopolizes that entire step, spiking every in-flight request's ITL to around 195ms. The p50 is low because most steps are the fast kind, and the p99 is high because of the occasional spike. Aggregate percentiles miss the story entirely, since the distribution has two modes and no requests actually experience the middle.

With chunked prefill, the distribution collapses to a single peak around 65ms. Prefill work is spread across many steps, so no single step is dramatically slower than the others.

The cost is TTFT. With chunked prefill, a 2048-token prompt does not finish prefilling until its chunks have been processed across multiple forward passes. At high concurrency, those passes are interleaved with many other requests, so each chunk has to wait in the queue before it runs. At c=64, median TTFT with chunked prefill is around 5 seconds versus 0.4 seconds without it. This is the same tradeoff described in the last post: `max_num_batched_tokens` controls how much prefill runs per step, and smaller values reduce ITL at the cost of TTFT. The configuration here uses a tight limit to make the effect clearly visible. It is also worth noting that closed-loop benchmarking keeps the queue permanently full, so each prefill chunk always has to wait behind a full batch of active decode requests. In a real system with variable traffic, the queue drains during quieter periods and the TTFT penalty would be smaller.

---

## What's next

The harness is built and the baseline is measured. Every optimization in the rest of the series will be measured against these numbers on the same hardware and workload, so we can isolate exactly what each feature buys.

Next up: prefix caching, where vLLM reuses KV cache entries across requests that share a common prefix, dramatically cutting TTFT for workloads with repeated system prompts or few-shot examples.
