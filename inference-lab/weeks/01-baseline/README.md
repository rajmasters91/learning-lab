### Session 2

## Hypothesis:
 how much GPU memory will nvidia-smi show in use once the vLLM server is idle, and why?

My assumption:  It will show 16GB (8B*2) in use since the model is loaded in VRAM. >>> WRONG

## Setup:

Card used: A100 80GB
GPU Used: GPU2
Model used:  Llama-3.1-8B-Instruct
vllm image used: vllm/vllm-openai:v0.31.0-ubuntu2404


docker run --gpus '"device=2"' \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=<redacted>\
  vllm/vllm-openai:v0.31.0-ubuntu2404\
  --model meta-llama/Llama-3.1-8B-Instruct


Command Breakdown: 
 --gpus '"device=2"' : assigns GPU with index 2 to the container.
 --ipc=host : shares host's shared memory space with container. Protect vLLM from crashing (due to pytorch shared-memory limits) during inference.
 -p 8000:8000: maps container port 8000 to host port 8000 to expose the webserver.
 -v ~/.cache/huggingface:/root/.cache/huggingface : stores the model in this location in host so that container can reuse the already downloaded model weights and doesnt need to pull it from scratch.
-e HF_TOKEN= : hugging face token passed as environment variable. mandatory for Llama 3.1, as it is a gated model i.e., requires licence acceptance from the model page.
vllm/vllm-openai:v0.31.0-ubuntu2404 : pulls openai vLLM docker image with tag "v0.31.0-ubuntu2404"
--model meta-llama/Llama-3.1-8B-Instruct : passed as vLLM entrypoint. says which HF model repo to load.



## Results:

nvidia-smi (74143 MiB used) shows vLLM grabs almost all VRAM upfront and pre-allocates KV cache, whether or not any request has arrived. 
Memory in use => budget vLLM was given != how big the model is.

# VLLM startup log: findings:

The dtype line (confirms BF16): ```dtype=torch.bfloat16```
How much memory the weights took, and how long loading took:  
```Model loading took 15.0 GiB memory and 5.420225 seconds```
```Loading weights took 0.87 seconds```
How much memory went to KV cache, and the number of tokens it can hold:  ```Available KV cache memory: 56.21 GiB``` and ```GPU KV cache size: 460,496 tokens```
The max_model_len it chose: ```max model len 131072```
The "maximum concurrency" line, if present: ```GPU KV cache size: 460,496 tokens, Maximum concurrency for 131,072 tokens per request: 3.51x``


# Actual logs: Run2 

(EngineCore pid=326) INFO 10-06 17:00:10 [default_loader.py:484] Loading weights took 0.87 seconds
(EngineCore pid=326) INFO 10-06 17:00:11 [model_runner.py:407] Model loading took 15.0 GiB memory and 5.420225 seconds

(EngineCore pid=326) INFO 10-06 17:00:33 [kv_cache_utils.py:2464] GPU KV cache size: 460,496 tokens, Maximum concurrency for 131,072 tokens per request: 3.51x

(EngineCore pid=326) INFO 10-06 17:00:41 [gpu_worker.py:955] Free memory on device (78.84/79.25 GiB) on startup. Desired GPU memory utilization is (0.92, 72.91 GiB). Actual usage is 15.26 GiB for consumed memory (weights + non-torch), 1.44 GiB for peak activation, and 0.23 GiB for CUDAGraph memory.

(EngineCore pid=326) INFO 10-06 17:00:33 [gpu_worker.py:692] Available KV cache memory: 56.21 GiB


# Analysis:


1. Memory Budget: 
gpu_memory_utilization=0.92 = 72.91 GiB  (0.92*79.25 GiB VRAM)
weights + Non-torch         = 15.26 GiB
Peak Activation Reserve     =  1.44 GiB

KV Cache = 72.91-15.26-1.44 = 56.21 GiB

Thus, KV Cache is the VRAM remaining after weights and working space (Peak Activation Reserve)
When we change the gpu_memory_utilization (ex: from 0.92 to 0.8), only KV cache is adjusted. Weights, Non-torch, Peak Activation Reserve remains the same. 

2. Why nvidia-smi shows used VRAM as 72.4 GiB (74143/1024), not 72.91 ? 
when idle, VRAM consumed = Weights(15.26) + KV cache (56.21) + CUDA graphs (0.23) =71.7 GiB + CUDA context of all GPU process (few 100 MiB usually, here 700MiB) = 72.4 GiB

1.44 GiB of Peak activation reserve headroom for forward pass and it is not used memory in idle. 

Hence, Budget != resident. 


3. 15.26 vs 15.0 GiB:
    Weights = 15.0 GiB.
    0.26 GiB “non-torch” = allocator + context overhead. 


4. KV cache per token:  
KV cache size (56.21 GiB) in KiB = (56.21*1024*1024) KiB
KV cache size in tokens = 460,496
KV cache per token = (56.21*1024*1024)/460,496 = 128KiB/tokens

5. Cold vs warm start. 
Run 1: model loading 22.07 s. 
Run 2: 5.42 s, with “Loading weights took 0.87 s”. Same files, same disk. 

What changed between runs? 
Linux page cache (buff/cache column of free -h) -> model is cached and hence run 2 took less time.
BUT,
Compilation stayed at ~15 s both times, so it isn’t cached across container restarts. 

Question for Week 11: Can a mounted cache directory fix that?

```
hostedai@dg05:~$ free -h
               total        used        free      shared  buff/cache   available
Mem:           503Gi        11Gi       377Gi       165Mi       118Gi       492Gi
Swap:             0B          0B          0B
hostedai@dg05:~$
```


6. Concurrency:

Total tokens / max model len =   460,496 ÷ 131,072 = 3.51

In worst case scenarion, Only 3–4 requests fit if each uses the full context. 
But, 1,000-token chat request would allow about 460 concurrent requests (460,496/1000) on the same cache. 

Note:
A100 card’s total VRAM as per nvidia-smi= 81,920 MiB ÷ 1024 = 80.0 GiB
vLLM log’s “79.25 GiB total” reports what CUDA can see after the driver reserves its share.


### Session 3:


## Summary

check "Session3_stream_run-gpu2-2026-10-11.txt" in 01-baseline/logs directory.

Run 1 (cache miss): first chunk at 0.182 s. That’s TTFT: ~140 ms of prefill for 2,065 tokens plus ~40 ms of fixed overhead.
```0.182  data: {"id":"chatcmpl-bdc514b73311a28f","object":"chat.completion.chunk","create```

Runs 2 and 3 (cache hit): first chunk at 0.042–0.047 s. The prefill vanished; what’s left is request handling, the last partial cache block, and the first decode step.
```0.042  data: {"id":"chatcmpl-a8b01a3009b56981","object":"chat.completion.chunk","create```

Cache miss - cache hit = ~140 ms

Decode speed was same in all runs: => Decode doesnt care about prefill.
i.e, from first content chunck to "[DONE]" took 0.72s 

Ex: run1: first chunk @ 0.182, "[DONE]" @ 0.908 = 0.72ms (0.908-0.182)
    run3: first chunk @ 0.047, "[DONE]" @ 0.769 = 0.72ms


To find the token count of a prompt, use the model’s own tokenizer only. Because, every model family has its own vocabulary => same text splits differently.



## Results:

1. Token Count: 
   
   Below field will be there in output of each query to the vLLM server. 
   
   Note: 
    1. We set max_tokens = 64 in the prompt query (check logs)
    2. Thumb rule: 1 token = ~5.3 characters (or) ~0.75 words
   
   "finish_reason":"length",  
   "usage": {  
     "prompt_tokens":45,
      "total_tokens":109,
      "completion_tokens":64, => output completed at 64 tokens as per max_tokens 
      },

    "finish_reason":"stop" => output completed within max_tokens limit
    "finish_reason":"length" => output terminated since it exceeded max_tokens
    "prompt_tokens" => tokens used by user prompt.
    "completion_tokens" => token consumed for query output = max_tokens 
    "total_tokens"  => prompt_tokens + completion_tokens
  
   The vLLM server exposes a "/tokenize" API endpoint. This can be used to identify the token consumption for a prompt. 

2. Short vs long, 3 runs each, with min_tokens forcing 64: 0.741 vs 0.762 s (cached) and 0.904 s (miss).


3. Predicted prefill 104 ms, measured ~140 ms; the gap is attention and overhead.


  Billion = 10^9 (1,000,000,000)
  Tera = 10^12 (1,000,000,000,000)

Prefill Compute = 2 operations * 8B parameters * prompt tokens

Long Prompt:
For 2000 tokens, how many TFLOPS of compute ?  its 2035 tokens incl wrapper/template overhead of 35 tokens.
 (2 x 8 x 10^9 x 2035)/(10^12) = 33 TFLOP 
  Note: FLOP is an amount of work; FLOPS (per second) is a rate.
At 312 TFLOPS/s for A100 card, 2000 token prompt takes ~104 ms (33/312) for prefill compute.

Short Prompt:
For a 10 token prompt , (2 x 8 x 10^9 x 45)/(10^12) = 0.72 TFLOPS
At 312 TFLOPS/s for A100 card, 10 token prompt takes 0.23 ms for prefill compute. 




4. Prefix caching discovered via hidden state; confirmed by editing one early word.

5. Decode: 11.5 ms/token ≈ 87 tokens/s ≈ 72% of the 120 tokens/s ceiling.

6. Streaming: TTFT 182 ms (miss) vs 45 ms (hit); decode rhythm unchanged.
