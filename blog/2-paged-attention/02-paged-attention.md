# Blog 1: PagedAttention

---

# PagedAttention


In the last post, we discussed how the naive approach to KV cache memory management wastes most of our GPU memory. Every request gets a pre-allocated contiguous block sized for the worst case, and as requests finish at different times, the available memory becomes fragmented into gaps that are too awkward for new requests to use. Thus, a production GPU would end up with most of its memory either idle inside active requests or stranded in unusable gaps between them.

Fortunately, this is not a new problem. vLLM borrows the solution from operating systems, which solved a nearly identical problem decades ago.

---

## Virtual Memory

Early operating systems allocated memory naively: each program claimed a contiguous chunk of physical memory. If a program needed 100 MB, and 100 MB happened to be free but scattered across many different gaps, the program was out of luck. As programs started and stopped, memory became increasingly fragmented, and the OS could not use free memory that was in one of the gaps.

The solution was *virtual memory* with *paging*. Instead of giving programs direct access to physical RAM, the OS gave each one a private virtual address space. From the perspective of the program, the memory was contiguous, but behind the scenes, the OS mapped the virtual address to wherever the data actually lived in physical RAM. Physical memory was split into fixed-size chunks called *pages* (usually 4 KB). A program's virtual address space was also split into *virtual pages* of the same size. The OS maintained a *page table* for each process, a mapping from virtual page number to the real physical page number.

The key insight was that with the page table, virtual pages did not need to map to contiguous pages in physical memory. A program's address space could thus be spread across physical RAM in any order while the OS hid the translation. Since every allocation was page-sized, there were no awkward gaps, as any free physical page could be used in any allocation request.

## PagedAttention for the KV cache

vLLM's **PagedAttention** applies the same idea to GPU memory. The KV cache for a request is no longer stored in one pre-allocated contiguous block. Instead, it is split into fixed-size *logical blocks*, where each block holds the keys and values for a fixed number of tokens (typically 16). These are the equivalents of virtual pages. Physical GPU memory is divided into *physical blocks* of the same size, pre-allocated at server startup. Instead of allocating a contiguous region per request, vLLM maintains a pool of available physical blocks and distributes them as needed.

Each request also gets a *block table*, the equivalent of a page table, which maps a logical block index to a physical block index in VRAM (GPU RAM). Logical block 0 might be at physical location 2 or 20 or 543. The request doesn't know or care, as the block table abstracts away the physical location entirely.

Figure 1 shows how a request's logical blocks map to physical blocks scattered across GPU memory, with the block table tracking where each logical block lives.

include block table image, cited
Figure from Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” arXiv:2309.06180, licensed under CC BY 4.0.

As a request generates tokens, it fills up its current logical block. When the block is full, vLLM allocates a new physical block from the available pool and adds an entry to the block table. When the request finishes, all held physical memory blocks are returned to the pool and made available to any other requests. 



## Computing attention across blocks

One may now wonder how attention can still be computed efficiently, given that all of the keys and values are scattered across non-contiguous physical memory. The answer is that vLLM implements a custom attention kernel that is aware of the block table. Before computing attention, the kernel uses the table to gather the necessary physical blocks in the correct order. The translation is then handled at the kernel level, meaning the data appears contiguous and the standard attention algorithm works without modification.


## The benefits

PagedAttention has a large impact on memory efficiency in practice. In the original vLLM paper, the authors measured that Orca, an earlier serving system with naive allocation, used only 20-40% of KV cache memory for actual tokens (depending on the configuration). With PagedAttention, the utilization rose to ~96%. On a GPU where KV cache capacity is the main limit on concurrency, a 2-4x improvement in memory utilization allows us to serve far more concurrent requests on the same hardware.

include usage image, cited
Figure from Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” arXiv:2309.06180, licensed under CC BY 4.0.

PagedAttention eliminates the two sources of memory waste that we discussed earlier: over-allocation and fragmentation.

- Over-allocation is eliminated because blocks are only allocated when the tokens actually arrive. A request that only uses 200 tokens never has memory reserved for tokens beyond that.

- Fragmentation is eliminated because every free block is the same size. When a request finishes and releases its blocks, they are immediately usable by any other active request. There are no longer any awkward, unusable gaps.


## Bonus: copy-on-write

The block table enables an additional capability called **copy-on-write** for shared memory. If two requests share an identical prefix, like a system prompt, their block tables can point to the same physical blocks for those prefix tokens. If one of these requests later needs to modify a shared block, vLLM creates a private copy of the block for the request and updates its block table. This is very similar to the copy-on-write that operating systems use for forked processes.

We will revisit this idea in later posts on optimizations like prefix caching (sharing blocks across requests) and speculative decoding (blocks are allocated speculatively and may be rolled back).


## What's next
PagedAttention gets us past the first roadblock of memory. We can now handle far more concurrent requests with the same hardware. However, actually scheduling them through the GPU efficiently is a separate problem.

A naive approach would just process one request at a time, and a slightly better one would wait for a batch to fill up before processing. Both of these solutions leave the GPU significantly underutilized. The next post covers vLLM's continuous batching and scheduling. We will look at how vLLM mixes incoming and in-progress requests to keep the GPU saturated, how the scheduler decides which requests to run at a given time, and how it handles the balance between throughput and latency. 