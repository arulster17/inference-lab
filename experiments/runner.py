import asyncio
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset

from benchmark.client import run_benchmark
from benchmark.workload import synthetic_workload, sharegpt_workload
from benchmark.metrics import compute_metrics

def run_experiment(config: dict) -> dict:
    # 1. load tokenizer
    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model)

    # 2. generate workload
    workload_str = config["workload"]
    num_requests = config["num_requests"]
    prompt_len = config["prompt_len"]
    if workload_str == "synthetic":
        requests = synthetic_workload(num_requests=num_requests, 
                                      prompt_len=prompt_len, 
                                      tokenizer=tokenizer)
    elif workload_str == "sharegpt":

        dataset = load_dataset(
            "json",
            data_files="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
            split="train"
        )


        requests = sharegpt_workload(num_requests=num_requests,
                                     tokenizer=tokenizer,
                                     dataset=dataset)
    
    else:
        raise ValueError(f"Unknown workload: {workload_str}")
        
    # 3. record start time, run benchmark, record end time
    base_url = config["base_url"]
    max_tokens = config["max_tokens"]
    concurrency = config["concurrency"]
    

    time_start = time.perf_counter()
    results = asyncio.run(run_benchmark(base_url=base_url,
                                        model=model,
                                        requests=requests,
                                        max_tokens=max_tokens,
                                        concurrency=concurrency))


    total_duration_s = time.perf_counter() - time_start

    # 4. compute metrics
    metrics = compute_metrics(results=results, total_duration_s=total_duration_s)


    # 5. return
    return {"config": config, "metrics": metrics}

def save_results(results: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(results, file)

def main():
    # hardcode config for now — we'll make this CLI-driven later
    config = {
        "base_url": "http://localhost:8000",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "workload": "synthetic",
        "num_requests": 100,
        "prompt_len": 512,
        "max_tokens": 256,
        "concurrency": 1,
    }
    results = run_experiment(config)
    save_results(results, "results/baseline_c1.json")

if __name__ == "__main__":
    main()