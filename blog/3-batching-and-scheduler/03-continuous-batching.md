# Blog 3: Continuous Batching and the Scheduler

---
    
# Continuous Batching and the Scheduler

In the previous post, we discussed how vLLM's PagedAttention system solved GPU memory fragmentation by splitting the KV cache into fixed-size blocks that could be scattered across physical memory. With PagedAttention, we can now handle far more concurrent requests in GPU memory at once. Unfortunately, memory capacity is only half of the problem. To keep the GPU utilized at all times, we need to decide which requests to group together and in what order to run them.

---

# Batching Basics

## Naive Batching

The straightforward approach to batching is to collect a group of requests, process them together until all of them finish, then start the next batch. This is much better than processing one request at a time, since the GPU can work on multiple requests in parallel. However, this approach runs into two big problems. First, requests arrive at different times. Since a naive strategy waits until it has enough requests to fill the batch, early requests are forced to wait, increasing their *time-to-first-token (TTFT)*. Second, requests have very different input and output lengths, so a naive batcher pads each request to match the longest, which wastes GPU compute. Furthermore, since a batch can't finish until its longest request is done, even more GPU compute goes to waste once shorter requests are finished.

Figure 1 shows four requests with different lengths arriving at different times. Request A arrives first but has to wait for 300 steps for the batch to fill up. Request D arrives last and generates the most tokens, keeping the batch active until step 500, long after the other three requests have finished.

insert static batch image here
caption: Figure 1: Naive batching with four requests of varying arrival times and generation lengths.

---

## Continuous Batching

**Continuous batching** solves both of these problems by scheduling at the iteration level rather than the request level. Instead of waiting for an entire batch to finish, it processes one forward pass of all requests in the batch at a time, after which completed requests are dropped and new requests are added. Furthermore, custom GPU kernels allow sequences of different lengths to be processed in the same forward pass, eliminating the need for padding. With this strategy, the GPU is now almost always doing useful work. Requests that finish free their slots immediately, and the slots are filled by the next available request rather than sitting idle.

Figure 2 shows the four requests arriving at the same staggered times. As each request finishes, its slot is freed immediately and filled by the next waiting request.

insert continuous batch image here
caption: Figure 2: Continuous batching with four requests of varying arrival times and generation lengths.


# The vLLM Scheduler


## Preemption

vLLM's scheduler uses a simple *first-come, first-served (FCFS)* priority queue, only admitting a new request when there is enough free memory for its initial KV cache blocks. However, since this check only covers a request's initial allocation and output lengths are not known in advance, memory pressure can build up as active requests keep generating tokens and expanding their KV caches. If available blocks run out, vLLM *preempts* the lowest-priority request (usually the most recent arrival), freeing its physical blocks to make room for the rest. The preempted request is handled using one of two strategies: 

- *Swap*: vLLM writes the request's KV cache blocks to CPU RAM and restores them later. This preserves the work done but costs memory bandwidth.
- *Recompute*: vLLM discards the KV cache and reruns prefill from scratch when the request is rescheduled. This avoids the bandwidth cost but wastes completed computation.

Both operations are costly and greatly spike the preempted request's latency, so the scheduler tries to avoid preemption by being conservative with request admission. vLLM V1 defaults to recompute over swap, as it has lower overhead in the V1 architecture.

## Chunked Prefill

Every request in the batch goes through two distinct phases:

- **Prefill**: the model processes the entire input prompt in a single forward pass. This step is compute-bound, since many tokens are processed in parallel, giving the GPU's arithmetic units a lot of work per step.
- **Decode**: the model generates one new token per forward pass, reading the full KV cache of all previous tokens on each step. This step is memory-bandwidth-bound, since the bottleneck is how fast the GPU can move data from memory to compute.

By default, the vLLM scheduler prioritizes prefills and doesn’t put prefill and decode requests in the same batch. This policy optimizes TTFT but results in slower *inter-token latency (ITL)* and inefficient GPU usage. Figure 3 shows a large prefill request taking priority in Pass 1, with decode requests unable to continue until the next pass. 

Figure 3: Without chunked prefill, prefill requests are prioritized, blocking requests that are mid-decode.
no chunks image
Caption: Figure 3: Without chunked prefill, prefill requests are prioritized, blocking requests that are mid-decode.


**Chunked prefill** solves this by limiting how many prefill tokens can enter the batch per step. The scheduler operates on a total token budget per forward pass (`max_num_batched_tokens`). At each step, vLLM first schedules all pending decode requests, then fills the remaining budget with prefill requests. If a waiting prefill request does not fit in the remaining token budget, vLLM chunks it and runs the appropriate chunk in the pass. With this system, decode requests are always prioritized, and the cost of a prefill request is spread across multiple steps instead of hitting one forward pass.

Figure 4 shows the same scenario with chunked prefill: decode requests are scheduled first, and the prefill is spread across multiple chunks.
chunks image
Caption: Figure 4: With chunked prefill, decode is scheduled first and prefill is split across passes.


Mixing compute-bound prefill chunks with memory-bandwidth-bound decode steps in the same pass also has the benefit of higher GPU utilization, as the GPU's compute units and memory are both kept busy in each pass.

The main tradeoff here is TTFT. A new request's prefill can now span multiple steps instead of being processed in one pass. We can tune `max_num_batched_tokens` to balance ITL and TTFT; smaller values reduce ITL and increase TTFT by limiting how much prefill can run alongside decode, while larger values decrease TTFT by processing more prefill tokens per step. The improvement to ITL is significant enough that chunked prefill is enabled by default in vLLM V1 whenever possible.


## What's Next?

We have now explored how vLLM addresses all three problems described in the first post: PagedAttention for memory, continuous batching for GPU utilization, and a scheduler with chunked prefill to manage latency. In the next post, we will put these concepts into practice: we will set up a vLLM instance on a cloud GPU, measure TTFT, ITL, and throughput, and build a baseline for the optimizations ahead.