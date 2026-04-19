# Blog 0: Setting Up vLLM and Serving Your First LLM

**Status:** Planned
**Thesis:** Get a real LLM running and serving requests in the cloud, from scratch.

---

## Outline
dg
### 1. Introduction
- Why self-host an LLM at all?
- What vLLM is and why it's the standard

### 2. Picking your cloud instance
- RunPod vs AWS/GCP/Azure — why bare GPU rental makes sense for this
- How to calculate VRAM requirements
- What we're using: RunPod A100 80GB

### 3. Setting up the instance
- Walk through `setup/install.sh`
- HuggingFace login and downloading Llama 3.1 8B
- Key gotchas (gated model, hf_transfer)

### 4. Starting the server
- Walk through `serving/configs/baseline.yaml` — what each flag does
- Running `serving/launch.py`
- Reading the startup logs

### 5. Sending your first request
- Walk through `scripts/test_request.py`
- Show the output — tokens streaming in real time

### 6. What's next
- The server works but we have no idea if it's fast or slow
- Next post: building a benchmark to measure exactly that
