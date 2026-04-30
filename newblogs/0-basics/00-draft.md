# Blog 0: LLM Inference Basics

**Status:** Draft
**Thesis:** Before understanding how to serve LLMs efficiently, you need to understand what inference actually does and why it creates hard problems at scale.

---

# LLM Inference Basics

When we use LLMs through an API, we send a request and tokens come back. A lot is hidden inside that abstraction. On the other side is a GPU, a model loaded into memory, a scheduler batching requests together, and a server orchestrating the entire process.

In this series, we will first understand the motivations behind vLLM, then explore various optimizations such as chunked prefill, prefix caching, and quantization. For this first post, we'll cover the foundations: how LLMs actually generate text, why the KV cache exists, and the three problems that make serving LLMs at scale difficult.

---

## How LLMs generate text

LLMs operate on tokens, not words. A token is roughly a word or word fragment, and the model's vocabulary is a fixed set of them. When you send a prompt, it gets tokenized into a sequence of integers before the model ever sees it.

The model processes this sequence through a forward pass: the input tokens go in, and a probability distribution over the entire vocabulary comes out. The model samples from this distribution to select the next token, appends it to the sequence, and runs another forward pass. This repeats until the model generates a stop token or hits a length limit.

This is called autoregressive generation, and the key property is that each new token depends on all the tokens that came before it. That dependency means generation is fundamentally sequential. You cannot generate token 10 until you have token 9, and you cannot generate token 9 until you have token 8. No matter how powerful your GPU is, you are always waiting on the previous step.

This sequential nature is what makes LLM inference slow relative to most GPU workloads, and it is the starting point for understanding why serving is hard.

---

## The KV cache

The attention mechanism is the core of how transformers process language. To understand why the KV cache exists, it helps to build an intuition for what attention is actually doing.

Think of attention as a soft database lookup. When processing a token, the model wants to gather relevant information from all the previous tokens in the sequence. Each token produces three things: a query, a key, and a value. The query represents what the current token is looking for. Each previous token's key represents what that token offers. The model compares the current query against every key to produce a set of relevance scores, then uses those scores to take a weighted average of the values. The result is a representation that blends information from across the sequence, weighted by relevance. This happens for every token, at every layer of the model, on every forward pass.

The problem is that during generation, the sequence grows by one token per step. Without optimization, every forward pass would recompute the keys and values for every token in the sequence from scratch. For a sequence that has grown to 512 tokens, generating token 513 requires recomputing 512 sets of keys and values. Generating token 514 requires recomputing 513, and so on. The compute cost grows quadratically with sequence length.

The fix is straightforward: cache the keys and values after computing them. The next time you need them, read from cache instead of recomputing. Queries do not need to be cached because you only ever need the query for the current token. Keys and values belong to every previous token and are needed on every subsequent step, so those are what you store. This takes the compute cost from quadratic to linear. Without it, the transformer would be unusably slow at any meaningful sequence length.

---

## The three problems

This is where vLLM comes in. vLLM is a serving system built specifically to handle the challenges that arise when running LLMs at scale. There are three core problems it needs to solve.

**Memory**

The KV cache is necessary, but it comes at a cost. The model weights are large but fixed: you load them once and they stay put. The KV cache is different: it grows throughout a request's lifetime and can only be freed once the request finishes. Every token in every active request has a cache entry that must stay in GPU memory until completion. For a request with a 512-token prompt generating 512 output tokens, that is 1024 tokens worth of cache. Multiply across many concurrent requests and you are holding tens of gigabytes of KV cache in VRAM, growing with every token generated. As we scale, the KV cache becomes the primary constraint on how many requests we can serve concurrently.

The naive way to manage this is to reserve a contiguous block of VRAM for each request upfront, sized for the maximum sequence length it might reach. This creates two problems. First, most requests end up shorter than their allocation, leaving a large portion of reserved memory sitting idle for the request's entire lifetime. Second, as requests complete and free their blocks at different times, the freed memory is scattered in chunks that are too awkward to reuse for new requests. This is fragmentation. Together, internal waste from over-allocation and external waste from fragmentation mean a significant portion of your VRAM is unavailable at any given time, directly capping how many requests you can serve simultaneously.

**Batching**

GPUs are massively parallel processors. A modern A100 has thousands of CUDA cores designed to execute operations simultaneously, and the way you exploit that parallelism is by giving the GPU a large amount of work to do at once.

Running a single LLM request at a time does not do that. A single request's forward pass uses only a fraction of the GPU's available compute: the arithmetic simply is not enough to keep all those cores busy. On a GPU processing one request at a time, you might sustain around 90 tokens per second. On the same GPU handling many concurrent requests efficiently, that number climbs into the thousands. Same hardware, an order of magnitude difference in throughput. The gap comes entirely from how well you are utilizing the parallelism the GPU offers.

The solution is batching: run multiple requests through the forward pass together. The GPU processes them in parallel, utilization goes up, and throughput improves dramatically.

But batching creates a tension with latency. Requests arrive continuously, at unpredictable intervals, with unpredictable lengths. If you wait to accumulate a large batch before processing anything, early requests sit idle while later ones trickle in. The longer you wait, the better your throughput but the worse your time to first token, which is the delay a user experiences before seeing any response at all. If you process every request the moment it arrives, latency is low but you are leaving most of the GPU unused. A serving system has to find the right point on that curve, and it has to do so continuously as traffic patterns shift.

**Scheduling**

Not all parts of request processing are the same. When a request first arrives, the model processes the entire input prompt in a single forward pass. This is called prefill. It is compute-bound, meaning the GPU's arithmetic units are the bottleneck. After prefill comes the decode phase, where the model generates one token per forward pass. Decode is memory-bandwidth-bound: each step reads the full KV cache for all active requests, and the bottleneck is how quickly the GPU can move data from memory to compute. While each decode step is fast, it runs once per output token, so there are potentially hundreds of decode steps per response. Prefill runs only once but can be slow for long prompts, since the entire input is processed in a single forward pass.

The scheduling challenge is that prefill and decode have different bottlenecks but share the same GPU. If a large prefill is batched together with requests that are mid-decode, the batch size increases significantly, each forward pass takes longer, and every decoding request in that batch sees higher per-step latency. Managing how to interleave prefill and decode across concurrent requests is what the scheduler is responsible for. Getting it wrong creates latency spikes that show up in tail metrics.

---

## What's next

These three problems each have a solution in vLLM. The next post covers PagedAttention, which addresses the memory problem by eliminating both over-allocation and fragmentation through a paging scheme borrowed from operating system memory management. After that, we'll look at continuous batching and the scheduler, which address the GPU utilization tension and the prefill/decode interference problem. Once those foundations are in place, we'll move to experiments and start measuring the effects of specific optimizations.
