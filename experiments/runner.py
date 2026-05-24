import asyncio
import json
import time
import argparse
import yaml

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
    return metrics

def save_config(config: dict, vllm_config_path: str, output_dir: str) -> None:
    config_path = Path(output_dir) / "config.json"
    if config_path.exists():
          return
    with open(vllm_config_path) as f:
        vllm_config = yaml.safe_load(f)
    combined = {
        "benchmark": {k: v for k, v in config.items() if k not in ("concurrency", "base_url")},
        "server": vllm_config
    }
    with open(config_path, "w") as f:
        json.dump(combined, f, indent=2)

def save_results(metrics: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(metrics, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vllm-config", type=str, required=True)
    args = parser.parse_args()

    config = {
        "base_url": "http://localhost:8000",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "workload": "synthetic",
        "num_requests": 4096,
        "prompt_len": 2048,
        "max_tokens": 512,
        "concurrency": args.concurrency
    }
    
    metrics = run_experiment(config)
    save_config(config, args.vllm_config, str(Path(args.output).parent))
    save_results(metrics, args.output)

if __name__ == "__main__":
    main()