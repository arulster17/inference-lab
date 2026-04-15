import yaml
import subprocess
import sys

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

def main():
    # get config path from sys.argv[1]
    # load it, build the command, print it, then subprocess.run it
    config_path = sys.argv[1]
    yaml_dict = load_config(config_path)
    command = build_command(yaml_dict)
    print(command)
    subprocess.run(command)


if __name__ == "__main__":
    main()

