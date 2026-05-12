# Blog 2: Continuous Batching and the Scheduler

---
    
# Continuous Batching and the Scheduler

In the previous post, we discussed how PagedAttention solved GPU memory fragmentation by splitting the KV cache into fixed-size blocks that can be scattered across physical memory. With PagedAttention, we can now handle far more concurrent requests in GPU memory at once. Unfortunately, memory capacity is only half the problem. To keep the GPU utilized at all times, we need to decide which requests to group together and in what order to run them.

---

## Naive Batching

The straightforward approach to batching is to collect a group of requests, process them together until all of them finish, then start the next batch. This is much better than processing one request at a time, since the GPU can work on multiple requests in parallel. However, this approach runs into two big problems. First, requests arrive at different times. Since a naive strategy waits until it has enough requests to fill the batch, early requests sit idle, increasing their *time-to-first-token* (**TTFT**). Second, requests can have very different input and output lengths, so a naive batcher has to pad the inputs and outputs of the requests to normalize the lengths, which wastes GPU compute and memory.

Suppose A, B, C, and D arrive in that order. A arrives first and waits 150 steps for the batch to fill — those 150 steps add directly to its TTFT. B waits 100 steps, C waits 50, and D arrives last and triggers the batch to start. Once processing begins, A generates 50 tokens, B generates 100, C generates 200, and D generates 500. A finishes quickly but its slot sits idle for 450 steps while D keeps running. Any request arriving after the batch starts has to wait for all 500 of D's steps before it can begin, also hurting TTFT.

[image]

---

## Iteration-Level Scheduling

**Continuous batching** fixes this by scheduling at the iteration level rather than the request level. Instead of waiting for an entire batch to finish, it operates on individual forward passes. After each forward pass, completed requests are immediately removed from the batch and new ones are added.

The result is that the GPU is always doing useful work. Fast requests that finish early free their slots right away, and those slots are filled by the next waiting request rather than sitting idle. A new request only has to wait for a single forward pass before it can be admitted, rather than waiting for an entire batch to complete. The vLLM paper reports 2-4x higher throughput compared to prior systems like FasterTransformer and Orca, largely because GPU utilization stays consistently high.

---

## Prefill vs. Decode

Continuous batching introduces a new complication. Requests in the active batch are in one of two phases:

- **Prefill**: the model processes the entire input prompt in a single forward pass. This step is compute-bound, since many tokens are processed in parallel, giving the GPU's arithmetic units a lot of work per step.
- **Decode**: the model generates one new token per forward pass, reading the full KV cache of all previous tokens on each step. This step is memory-bandwidth-bound, since the bottleneck is how fast the GPU can move data from memory to compute.

These two phases have very different resource profiles. A prefill step for a long prompt does a large amount of arithmetic in a single pass. A decode step does very little arithmetic but needs to touch a lot of memory.

When both are mixed in the same batch, prefill dominates. For a decode request that is mid-generation, a forward pass that would normally complete in a few milliseconds can take significantly longer because a newly admitted request's long prompt is being processed in the same step. The decode request's next token is delayed, causing a spike in ITL. Ongoing decode requests pay a "prefill tax" every time a new request joins the batch.

This creates a real tension. Starting new requests quickly keeps TTFT low, which is good for the user experience of those requests. But starting them means running prefill alongside decode, which inflates ITL for everyone already in the batch. A good scheduler has to manage this tradeoff continuously.

---

## The vLLM Scheduler

vLLM's scheduler uses a *first-come, first-served (FCFS)* priority queue. Incoming requests are queued in order of arrival. At each step, the scheduler decides which requests run in the next forward pass based on a memory budget.

Requests already in the decode phase have priority. They are holding physical KV cache blocks and making progress toward completion. Preempting them to admit new prefill requests would waste the work already done and introduce unnecessary latency spikes. New requests waiting in the queue are only admitted when there is enough free memory to store their initial KV cache blocks.

Occasionally, a spike in concurrent traffic can push memory pressure high enough that the scheduler can't fit all active decode requests at once. In this case, vLLM *preempts* the lowest-priority request, freeing its physical blocks to make room for the rest. It has two strategies for handling the preempted request:

- *Swap*: write the request's KV cache blocks from GPU memory to CPU RAM, then swap them back in later when memory becomes available. This preserves the work done so far but costs CPU-GPU bandwidth.
- *Recompute*: discard the KV cache entirely and rerun the prefill from scratch when the request is rescheduled. This wastes the computation already done but avoids the bandwidth cost of swapping.

Both strategies carry a cost. The scheduler tries to avoid preemption altogether by being conservative about how many requests it admits at each step. In practice, preemption is relatively rare under steady traffic, but it is a necessary safety valve for bursty workloads.

---

## The Throughput/Latency Tradeoff

Even with continuous batching, there is an irreducible tradeoff between throughput and latency.

To maximize throughput, we want as many requests in flight as possible. More concurrent requests means more tokens generated per second. But more concurrent requests also means longer queues, higher memory pressure, and potentially more prefill-decode interference, all of which push TTFT and ITL higher.

To minimize latency, we want to start new requests immediately and keep the active batch small. But a smaller batch means the GPU is doing less work per forward pass, and throughput falls.

The prefill tax sharpens this tradeoff. A straightforward fix is to break up long prefill steps into smaller chunks, each processed in a separate forward pass, so that no single prefill step can dominate a batch. This caps the interference between new and in-progress requests without requiring us to delay new ones entirely. This is the idea behind *chunked prefill*, which we will cover in a later post.

---

## What's next

We have now covered the three core ideas behind vLLM: PagedAttention for memory efficiency, and continuous batching plus scheduling for compute utilization. Before going further into optimizations, the next post puts all of this into practice. We will set up a vLLM instance, run a proper benchmark, and measure TTFT, ITL, and throughput at different concurrency levels to build a concrete baseline for what the system actually looks like in numbers.
