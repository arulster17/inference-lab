import numpy as np
from benchmark.client import RequestResult


def compute_metrics(results: list[RequestResult], total_duration_s : float) -> dict:

    # only use sucessful results for latency stats
    successful = [r for r in results if r.error is None]
    ttft_results = np.array([r.ttft_s for r in successful])
    ttft_p50 = float(np.percentile(ttft_results, 50))
    ttft_p95 = float(np.percentile(ttft_results, 95))
    ttft_p99 = float(np.percentile(ttft_results, 99))

    all_itls = np.concatenate([r.itl_s for r in successful if r.itl_s])

    if len(all_itls) == 0:
      itl_p50 = itl_p95 = itl_p99 = None
    else: 
        itl_p50 = float(np.percentile(all_itls, 50))
        itl_p95 = float(np.percentile(all_itls, 95))
        itl_p99 = float(np.percentile(all_itls, 99))

    total_requests = len(results)
    total_output_tokens = sum([r.output_tokens for r in results])

    throughput_rps = total_requests / total_duration_s
    throughput_tps = total_output_tokens / total_duration_s

    successful_requests = len(successful)
    error_rate = 1- (successful_requests / total_requests)

    return {
        "ttft_p50" : ttft_p50,
        "ttft_p95" : ttft_p95,
        "ttft_p99" : ttft_p99,
        "itl_p50" :  itl_p50,
        "itl_p95" :  itl_p95,
        "itl_p99" :  itl_p99,
        "throughput_rps" : throughput_rps,
        "throughput_tps" : throughput_tps,
        "error_rate" : error_rate,
        "total_requests" : total_requests,
        "successful_requests" : successful_requests
    }
