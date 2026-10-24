## Session 1: Prepare the GPU (about 2 hours)

Use nvidia-smi -q to find whether the card is SXM or PCIe, plus its memory, driver, and CUDA versions.
Enable persistence mode and lock the clocks. Look up why both matter for repeatable numbers.
Confirm MIG is off.
Install the NVIDIA Container Toolkit and prove that a container can see the GPU.



## Session 2: get the model served on GPU 2 and read what the startup log tells you.

Before you start

Check Hugging Face: is Llama-3.1-8B-Instruct access approved? If yes, create a read token. If not, switch to Qwen/Qwen2.5-7B-Instruct and we update the doc; don't wait on it.
Pick a vLLM image tag from Docker Hub (vllm/vllm-openai) and write it down. Not latest: a pinned tag is what makes Week 8 comparable to Week 1.

The run command, shape only. Look up each flag in the vLLM docs so you know what it does rather than copying:

docker run --gpus '"device=2"' \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=... \
  vllm/vllm-openai:<tag> \
  --model meta-llama/Llama-3.1-8B-Instruct

Three things to understand before running it:

-v for the cache. Without it, 16 GB downloads again every time the container restarts.
--ipc=host. vLLM uses shared memory between processes; the Docker default is too small.
No tuning flags. This is the baseline. Defaults are the point.

Write the prediction first in weeks/01-baseline/README.md: how much GPU memory will nvidia-smi show in use once the server is idle, and why? Hint: vLLM has a default called gpu_memory_utilization. Look up its value.

While it loads, watch two windows: the container log, and watch -n1 nvidia-smi -i 2.

In the startup log, find and record:

The dtype line (confirms BF16)
How much memory the weights took, and how long loading took
How much memory went to KV cache, and the number of tokens it can hold
The max_model_len it chose
The "maximum concurrency" line, if present

Save the whole log to weeks/01-baseline/logs/vllm_startup.log and the exact run command to weeks/01-baseline/config.yaml.

Then: compare idle memory use to your prediction. The gap, if any, is your first finding.

Bring back the five log lines and the nvidia-smi reading.




## Session 3: talk to the server

Goal: see tokens, prefill and decode with your own eyes before the benchmark tool hides them behind summary numbers.

1. Is it there?
curl http://localhost:8000/v1/models should list the model.

2. One chat request. POST to /v1/chat/completions with a JSON body containing model, messages (one user message) and max_tokens: 64. Look at the usage block in the response: prompt_tokens, completion_tokens, total_tokens. Paste the prompt into a tokenizer tool (Hugging Face has one on the model page) and check the count matches.

3. Write the prediction, then test it. Two requests, both max_tokens: 64:

Short prompt: ~20 tokens
Long prompt: ~2,000 tokens (paste a few paragraphs of anything)

Predict: which is slower, by roughly how much, and which phase takes the extra time? Then time both with time curl ....

4. See the phases separately. Add "stream": true to the body. The response arrives as chunks: the pause before the first chunk is TTFT (prefill); the rhythm of the chunks after it is decode. Run the long prompt streaming and watch where the wait is.

5. Peek at the metrics. curl http://localhost:8000/metrics | grep -E "prompt_tokens|generation_tokens|time_to_first_token". This is what Grafana will read in Week 2.

Record in the Week 1 README under Results: the usage numbers, the two timings, your prediction vs what happened, and one sentence on what streaming showed you.

Check your understanding afterwards: why did the long prompt cost extra time in one phase but not the other?


