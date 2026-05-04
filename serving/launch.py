import yaml
import subprocess
import sys
import time
import requests

def load_config(path: str) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)

def build_command(config: dict) -> list[str]:
    # returns something like: ["vllm", "serve", "meta-llama/...", "--dtype", "float16", ...]
    command = ["vllm", "serve", config['model']]
    for key,val in config.items():
        if key == 'model':
            continue
        command.append("--" + key)
        command.append(str(val))
    
    return command

def wait_for_server(base_url: str, timeout_s: int = 300):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health")
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass # keep waiting
        time.sleep(2)
    
    raise TimeoutError(f"Server did not become ready within {timeout_s}s")

def send_warmup_request(base_url: str, model: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello World!"}],
        "max_tokens": 5,
        "stream": False
    }
    requests.post(f"{base_url}/v1/chat/completions", json=payload)
    print("Warmup complete")

def main():
    # get config path from sys.argv[1]
    # load it, build the command, print it, then subprocess.run it
    config_path = sys.argv[1]
    config = load_config(config_path)
    command = build_command(config)
    print(command)


    # we want to test the health, so use popen instead of run for nonblocking

    base_url = f"http://localhost:{config['port']}"
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    wait_for_server(base_url)
    send_warmup_request(base_url, config["model"])

    # make sure program doesn't exit
    process.wait()


if __name__ == "__main__":
    main()

