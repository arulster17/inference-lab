# Blog 3: Continuous Batching and the Scheduler

---
    
# Continuous Batching and the Scheduler

In the previous post, we discussed how PagedAttention solved GPU memory fragmentation by splitting the KV cache into fixed-size blocks that can be scattered across physical memory. With PagedAttention, we can now handle far more concurrent requests in GPU memory at once. Unfortunately, memory capacity is only half the problem. To keep the GPU utilized at all times, we need to decide which requests to group together and in what order to run them.

---

## Naive Batching

The straightforward approach to batching is to collect a group of requests, process them together until all of them finish, then start the next batch. This is much better than processing one request at a time, since the GPU can work on multiple requests in parallel. However, this approach runs into two big problems. First, requests arrive at different times. Since a naive strategy waits until it has enough requests to fill the batch, early requests are forced to wait, increasing their *time-to-first-token* (**TTFT**). Second, requests have very different input and output lengths, so a naive batcher pads each request to match the longest, which wastes GPU compute. Furthermore, since a batch can't finish until its longest request is done, even more GPU compute goes to waste once shorter requests are finished.

Suppose A, B, C, and D arrive in that order. A arrives first and waits 300 steps for the batch to fill — those 300 steps add directly to its TTFT. B waits 200 steps, C waits 100, and D arrives last and triggers the batch to start. Once processing begins, A generates 50 tokens, B generates 100, C generates 200, and D generates 500. A finishes quickly but its slot sits idle for 450 steps while D keeps running. Any request arriving after the batch starts has to wait for all 500 of D's steps before it can begin, also hurting TTFT.

[image]

---

## Continuous Batching

**Continuous batching** solves both of these problems by scheduling at the iteration level rather than the request level. Instead of waiting for an entire batch to finish, it processes one forward pass of all requests in the batch at a time, after which completed requests are dropped and new requests are added. Furthermore, custom GPU kernels allow sequences of different lengths to be processed in the same forward pass, eliminating the need for padding. With this strategy, the GPU is now almost always doing useful work. Requests that finish free their slots immediately, and the slots are filled by the next available request rather than sitting idle.

Figure 2 shows the same four requests. As each request finishes, its slot is freed immediately and filled by the next waiting request — no idle time, no queueing delays.

[image]

---

## The Scheduler and Preemption

vLLM's scheduler uses a simple *first-come, first-served (FCFS)* priority queue, only admitting a new request when there is enough free memory for its initial KV cache blocks. However, since this check only covers a request's initial allocation and output lengths are not known in advance, memory pressure can build up as active requests keep generating tokens and expanding their KV caches. If available blocks run out, vLLM *preempts* the lowest-priority request — usually the most recent arrival — freeing its physical blocks to make room for the rest. The preempted request is handled using one of two strategies: *swap*, which writes its KV cache blocks to CPU RAM and restores them when it is rescheduled, preserving prior work at the cost of CPU-GPU bandwidth; or *recompute*, which discards the KV cache and reruns prefill from scratch, avoiding the bandwidth cost but wasting prior computation. Both operations are wasteful and greatly spike the preempted request's latency, so the scheduler tries to avoid preemption by being conservative with request admission. vLLM defaults to recompute — in v1's architecture, recomputation has lower overhead than the CPU-GPU transfer cost of swap.

---

## Chunked Prefill

Every request in the batch moves through two distinct phases: **prefill**, where the model processes the entire input prompt in a single forward pass, and **decode**, where it generates one new token per forward pass while reading the full KV cache of all previous tokens. These phases have fundamentally different resource profiles — prefill is compute-bound, processing many tokens in parallel; decode is memory-bandwidth-bound, with the bottleneck being how fast the KV cache can be read each step. When both run in the same forward pass, prefill dominates — a newly admitted request's prompt can slow the entire pass, delaying the next decode token for every other request and spiking ITL.

**Chunked prefill** solves this by limiting how many prefill tokens can enter the batch per step. The scheduler operates on a total token budget per forward pass (`max_num_batched_tokens`, default 512). At each step, it first schedules all pending decode requests, then fills the remaining budget with prefill tokens. If a waiting request's full prefill does not fit in the remaining budget, vLLM automatically chunks it — only the fitting portion runs this step, and the rest continues in subsequent steps. This keeps the forward pass time predictable: decode is never blocked by a large prefill, and no single prefill can dominate the batch.

Mixing compute-bound prefill chunks with memory-bound decode steps in the same forward pass also improves GPU utilization — the compute units and memory subsystem are both kept busy rather than one sitting idle while the other works.

The main tradeoff is TTFT — a new request's prefill now spans multiple steps rather than completing in one pass. `max_num_batched_tokens` controls this balance: smaller values reduce ITL by limiting how much prefill runs alongside decode; larger values improve TTFT by processing more prefill tokens per step. Chunked prefill is enabled by default in vLLM v1 whenever possible.

---

## What's next

With this post, we have now addressed all three problems from the first post: PagedAttention for memory, continuous batching to keep the GPU utilized, and a scheduler with chunked prefill to manage the prefill/decode tension and keep latency predictable. The next post puts this into practice — we will set up a vLLM instance, run a benchmark, and measure TTFT, ITL, and throughput at different concurrency levels to build a concrete baseline.
