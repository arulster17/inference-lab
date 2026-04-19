"""
Smoke test — send one streaming request to a running vLLM server.
Usage: python scripts/test_request.py --base-url http://localhost:8000
"""

import argparse
from openai import OpenAI

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "Explain what a large language model is in 3 sentences."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True
    )

    for chunk in stream:
          token = chunk.choices[0].delta.content
          if token:
              print(token, end="", flush=True)
    
    print()

    pass


if __name__ == "__main__":
    main()
