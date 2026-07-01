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

We run the sweep with 2048-token prompts and compare chunked prefill on versus off. We use longer prompts because the effect of chunked prefill is only visible when the prefill step is large relative to the decode step and `max_num_batched_tokens`. We also choose c=64, as it is at the knee of the throughput curve and is a realistic operating point.


itl 64 histogram image here /analysis/chunked_2048/itl_histogram_64.png

Without chunked prefill, the distribution is *bimodal*. Most decode steps are fast pure-decode steps clustered around 25 ms, but whenever a 2048-token prefill arrives, it monopolizes THE entire step and spikes every in-flight request's ITL to around 200 ms. This makes ITL very unpredictable in production systems, and our previous p50/p95/p99 type measurements would be unable to capture this behavior. 

With chunked prefill, the distribution collapses to a single peak around 65 ms. Prefill work is spread across many steps, so no single step is dramatically slower than the rest. This matches our prediction from the previous post: chunked prefill trades a few tall ITL spikes for a higher, more stable floor.

The cost is TTFT. With chunked prefill, a 2048-token prompt does not finish prefill until its chunks have been processed across several forward passes. At high concurrency those passes are interleaved with many other requests, meaning each chunk waits in the queue before it runs. At c=64, the median TTFT is around 5 seconds with chunked prefill vs ~0.4 seconds without it. `max_num_batched_tokens` controls how much prefill runs per step, and the tight limit of 512 used here makes the effect on ITL and TTFT clearly visible.

ttft image

Further up the sweep the TTFT numbers look even worse, as they reach ~30 seconds at c=128 and ~65 seconds at c=256, but this is due to saturation rather than the chunked prefill. The tight 512-token budget and the 4x larger prompts bring the throughput plateau down to ~700 tokens per second at just c=32, meaning the higher concurrencies are already running well past saturation, meaning their TTFT is dominated by the waiting time in the queue.

throughput img

While the TTFT numbers look extremely bad, it's important to keep the following caveats in mind:
- The tight max_num_batched_tokens=512 is chosen to exaggerate the effect of chunked prefiil. A realistic value like 2048 or 4096 would let prefill finish in fewer chunks and lower the TTFT.
- Closed-loop benchmarking keeps the queue permanently full. Real-life traffic comes in bursts, which would give pending prefills time to catch up.

---

## What's next

We've now built a test harness and measured our first optimization. Every optimization in the rest of the series will be measured with the same harness, hardware, and workload, meaning we can isolate the benefits of each new feature.

In the next blog, we'll look at speculative decoding, where a small draft model proposes tokens that the main model verifies in a single pass, potentially allowing the model to advance several tokens in one step. We'll see why this helps most at low concurrencies but can backfire at higher ones.
