import random
from datasets import load_dataset

def synthetic_workload(
    num_requests: int,
    prompt_len: int,        # tokens
    tokenizer,
) -> list[tuple[str, int]]:
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    prompt = " ".join(random.choices(words, k=prompt_len))
    enc = tokenizer.encode(prompt)
    cut = enc[:prompt_len]
    dec = tokenizer.decode(cut)

    return [(dec, len(cut)) for _ in range(num_requests)]
    

def sharegpt_workload(
    num_requests: int,
    tokenizer,
) -> list[tuple[str, int]]:
  
  
    
    ds = load_dataset("json",
        data_files="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train")
    
    request_data = random.sample(ds, int(num_requests * 1.2)) # give some buffer in case we skip some chats
    results = []
    for data in request_data:
        if len(results) == num_requests:
            break
        convs = data['conversations']
        first_human = next((c for c in convs if c['from'] == 'human'), None)
        if first_human is None:
            continue

        text = first_human['value']
        results.append((text, len(tokenizer.encode(text))))

    return results
sharegpt_workload(0,None)