import asyncio
import time
import json
from dataclasses import dataclass

import aiohttp

@dataclass
class RequestResult:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    itl_s: list[float]
    total_latency_s: float
    error: str | None = None

async def send_streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    max_tokens: int,
    request_id: str,
) -> RequestResult:
    payload = {
        "model": model,          # we'll make this configurable later
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t_start = time.perf_counter()
    t_last = None
    ttft_s = 0.0
    itl_s = []
    output_tokens = 0
    first = True
    try:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
            async for line in resp.content:
                lc = line.decode("utf-8").strip()
                if not lc.startswith("data: ") or lc == "data: [DONE]":
                    continue
                data = json.loads(lc[6:])
                if data["choices"][0]["delta"].get("content", ""):
                    output_tokens += 1
                    if first:
                        t_now = time.perf_counter()
                        ttft_s = t_now - t_start
                        t_last = t_now
                        first = False
                    else:
                        t_now = time.perf_counter()
                        itl_s.append(t_now - t_last)
                        t_last = t_now

        total_latency_s = time.perf_counter() - t_start
        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_s=ttft_s,
            itl_s=itl_s,
            total_latency_s=total_latency_s,
            error=None
        )
    except Exception as e:
        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_s=ttft_s,
            itl_s=itl_s,
            total_latency_s=time.perf_counter() - t_start,
            error=str(e)
        )


async def run_benchmark(
      base_url: str,
      model: str,
      requests: list[tuple[str, int]],
      max_tokens: int,
      concurrency: int,
  ) -> list[RequestResult]:
    
    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(prompt: str, prompt_tokens: int, request_id: str) -> RequestResult:
        async with sem:
            return await send_streaming_request(session=session, 
                                                base_url=base_url,
                                                model=model,
                                                prompt=prompt, 
                                                prompt_tokens=prompt_tokens, 
                                                max_tokens=max_tokens, 
                                                request_id=request_id)
        
    
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(pr[0], pr[1], f"req-{req_id}") for req_id, pr in enumerate(requests)]
        results = await asyncio.gather(*tasks)
        return results
        