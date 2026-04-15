import random
from transformers import PreTrainedTokenizerBase
from datasets import Dataset

def synthetic_workload(
    num_requests: int,
    prompt_len: int,        # tokens
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[str, int]]:
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    prompt = " ".join(random.choices(words, k=prompt_len))
    enc = tokenizer.encode(prompt)
    cut = enc[:prompt_len]
    dec = tokenizer.decode(cut)

    return [(dec, len(cut)) for _ in range(num_requests)]
    

def sharegpt_workload(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    ds: Dataset,
) -> list[tuple[str, int]]:
  
    
    indices = random.sample(range(len(ds)), int(num_requests * 1.2)) # give some buffer in case we skip some chats
    request_data = ds.select(indices)
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