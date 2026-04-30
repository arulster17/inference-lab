# Blog 0: LLM Inference Basics

**Status:** Draft
**Thesis:** Before understanding how to serve LLMs efficiently, you need to understand what inference actually does and why memory becomes the binding constraint.

---

## Outline

### 1. Introduction
- What this series is about
- Why understanding inference is the foundation for everything else

### 2. How LLMs generate text
- The forward pass: input tokens in, probability distribution out
- Autoregressive generation: why we generate one token at a time
- Why this is slow by nature

### 3. The KV cache
- What the attention mechanism computes
- Why recomputing attention for every token is wasteful
- The KV cache as the solution: store and reuse key-value pairs
- The cost: KV cache grows with every token generated

### 4. Why memory becomes the constraint
- Weights are large but fixed (16GB for Llama 3.1 8B in FP16)
- KV cache is dynamic and grows with sequence length and concurrency
- At scale, KV cache pressure is what limits how many requests you can serve
- This is the problem the rest of the series is about solving

### 5. What's next
- The first problem to solve: managing KV cache memory efficiently
- Next post: PagedAttention

---

## Draft

---

# LLM Inference Basics

When we use LLMs, we usually do so through an API, like OpenAI's API for GPT models or Anthropic's API for Claude, and output tokens are streamed back. A lot of steps get hidden in this abstraction. On the other side of the API is a GPU, a model loaded into memory, a scheduler batching requests together, and a server orchestrating the entire process.


In this series, we will first learn about vLLM and understand the motivation behind its features, then explore various optimizations such as prefix caching, quantization, speculative decoding, and more. This first post will focus on the foundations of LLMs, in particular how they generate text, what a KV cache is and why we need it, and problems that arise when we try to run LLMs at scale.

---

## How LLMs generate text

LLMs don't work directly with words. Instead, they use tokens, which are usually words or subwords (like "ing" or "mis"). A model is trained with a fixed set of them, called a *vocabulary*. When you send a prompt, a *tokenizer* converts the text into a sequence of tokens, which are then converted to their corresponding integers. 

Consider as an example the sequence "The hyperparameters weren't tuned properly". OpenAI's tokenizer converts this to ["The", "hyper", "parameters", "weren't", "tuned", "properly"], then converts the tokens to their token IDs, giving us [976, 22725, 24021, 52533, 52549, 13425]. This sequence of numbers is the input to the model.

The model processes this sequence with a forward pass, where the input tokens go in, and a probability distribution over the entire vocabulary comes out. The model samples from this distribution to pick the next token, adds it to the sequence, and then runs another forward pass. This continues until the model generates a STOP token or hits a length limit. This process is known as *autoregressive generation*. An important property of this method is that every new token depends on all tokens before it. Thus, generation must be sequential, as we can't generate token 10 until we have token 9, and we can't generate token 9 until we have token 8. The sequential nature of LLM inference means that a powerful GPU alone is not sufficient for a serving system.


## The KV cache

Most modern LLMs are based on the transformer architecture, the core of which is the attention mechanism. Attention is basically a soft lookup. When we process a token, the model wants to gather relevant information from all previous tokens to build a better output distribution. Each token then produces three things:

- Query: what the token is looking for.
- Key: what the token offers.
- Value: what the token contributes to the final representation.

When the model is processing a token, it compares its query to the keys of all past tokens to produce a set of relevance scores (or how much "attention" the current token should pay to a certain past token). With these attention scores, we take a weighted average with the corresponding values to get a representation that blends information from across the sequence.

The problem with this system arises during inference, as the sequence grows by one token in every step. Without caching, every forward pass would recompute the keys and values for all tokens in the sequence, meaning the compute cost would grow quadratically. The solution to this is to cache the keys and values for a token after computing them, as what a token will offer and contribute will not change as we process future tokens. We don't need to cache the queries, as they are only ever needed once: during the processing of the corresponding token. Since the keys and values of past tokens are needed for every subsequent step, storing them takes us from quadratic to linear time. Without this optimization, the transformer model would be unusably slow.

---

# From Running to Serving

We now understand the basics of running an LLM: an input sequence of tokens comes in, a forward pass outputs a new token to add to the sequence, and repeat until done. Running a model like this in isolation is straightforward. However, to serve the model, we need to be able to handle many concurrent users, keep GPU utilization high, and deliver fast responses. Before we can achieve this, three major roadblocks stand in our way.

## Memory

The KV cache is necessary, but it comes with a high memory cost. Every token in every request has a KV cache entry that needs to stay in GPU memory until the request is finished. Furthermore, each layer in an LLM has multiple attention heads, which each have their own keys and values for each token. For a request with prompt length 512 and output length 512, we get 1024 tokens worth of KV cache. When we combine the sequence length, the large number of attention heads, and the potential concurrent requests that our system needs to handle, we need to hold tens of gigabytes of KV cache in GPU memory. Model weights are large, but they take up a fixed amount of memory after we load them. The KV cache for a request grows throughout its lifetime and can only be freed once the request finishes. As we scale, the KV cache is a much bigger GPU memory sink, making it the primary constraint on how many requests we can serve concurrently.

The naive way to handle this is to reserve a contiguous block of VRAM for each request ahead of time, sized for the maximum possible sequence length. This creates two problems. First, many requests end far before the token limit, meaning a large portion of allocated memory is sitting idle for the request's entire lifetime. Second, as requests finish processing and free their blocks, the available memory for new requests can be scattered into various chunks that are too awkward to be used. The combination of internal waste from *over-allocation* and external waste from *fragmentation* means that we are using far more VRAM than our requests actually need, severely restricting how many requests we can serve concurrently.

## Batching

A GPU's main strength is its massive parallelism. A modern A100 has thousands of CUDA cores that are designed to execute operations simultaneously, and to exploit this parallelism, we need to send the GPU a large amount of work to do at once. Processing a single LLM request does not achieve this. A single request's forward passes use only a small fraction of the available compute, as there is simply not enough work to be done across all of the cores. A GPU processing one request at a time can reach around 90 tokens per second, but the same GPU handling multiple concurrent requests can output thousands of tokens in the same time. The gap comes down to how well we can use the parallelism offered by the GPU.

The solution is batching, where we run multiple requests through the forward pass together. The GPU processes them in parallel, and our *throughput* improves significantly. The tradeoff here is *latency*. Requests can arrive at any time and have unpredictable lengths. If we wait to accumulate a large batch before starting processing, early requests can sit idle while we wait for the batch to fill up. While the throughput improves as we wait, the time-to-first-token (**TTFT**) for earlier requests suffers, and the corresponding user experiences a longer delay before receiving any response. Conversely, if we process requests the moment they arrive, latency and TTFT are low, but a large part of the GPU remains unused. A good serving system needs to find a balance between latency and utilization and do so continuously to match traffic patterns. 

## Scheduling

There are multiple phases to processing a request. When it first arrives, the model needs to process the entire input prompt in a forward pass. This is called the *prefill* phase. This step is compute-bound, as the GPU's arithmetic units are the bottleneck. After prefill comes the *decode* phase, where the model generates one token per forward pass. This step is memory-bandwidth-bound, as each step needs to read the full KV cache, and the bottleneck is how quickly the GPU can move the information from memory to compute. While each decode step is fast, it runs once per output token, meaning there are potentially hundreds of decode steps per response. In contrast, prefill only runs once but can be slow for longer prompts, as the entire input gets processed in one forward pass.

The challenge with scheduling is that the prefill and decode phases have different resource needs but share the same GPU. If a large prefill step is run alongside requests that are mid-decode, the entire batch gets slowed down, and the inter-token latency (**ITL**) of the decoding requests spikes. A good serving system needs a *scheduler* that can interleave prefill and decode steps across requests to achieve predictable latency and avoid unpredictable spikes.

---

# What's next

We need to tackle all three of these problems to serve an LLM in a production environment. vLLM is a high-performance inference engine that solves these issues and more, and it has become the standard serving engine in the industry. The next few posts will explore how vLLM solves each of these problems. After that, we will explore further optimizations to serving engines and measure their effects.
