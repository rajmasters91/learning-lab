1. how much GPU memory will nvidia-smi show in use once the vLLM server is idle, and why?


My assumption:  It will show 16GB (8B*2) in use since the model is loaded in VRAM. >>> WRONG

Result: see nvidia-smi below. 
vLLM grabs almost all VRAM front and pre-allocates KV cache, whether or not any request has arrived. 
Memory in use => budget vLLM was given, NOT how big the model is.

As per nvidia-smi, 74143 MiB was used at idle. 




VLLM startup log: findings:

The dtype line (confirms BF16): ```dtype=torch.bfloat16```
How much memory the weights took, and how long loading took:  
```Model loading took 15.0 GiB memory and 5.420225 seconds```
```Loading weights took 0.87 seconds```
How much memory went to KV cache, and the number of tokens it can hold:  ```Available KV cache memory: 56.21 GiB``` and ```GPU KV cache size: 460,496 tokens```
The max_model_len it chose: ```max model len 131072```
The "maximum concurrency" line, if present: ```GPU KV cache size: 460,496 tokens, Maximum concurrency for 131,072 tokens per request: 3.51x``



Actual logs: Run2 

(EngineCore pid=326) INFO 10-06 17:00:10 [default_loader.py:484] Loading weights took 0.87 seconds
(EngineCore pid=326) INFO 10-06 17:00:11 [model_runner.py:407] Model loading took 15.0 GiB memory and 5.420225 seconds

(EngineCore pid=326) INFO 10-06 17:00:33 [kv_cache_utils.py:2464] GPU KV cache size: 460,496 tokens, Maximum concurrency for 131,072 tokens per request: 3.51x

(EngineCore pid=326) INFO 10-06 17:00:41 [gpu_worker.py:955] Free memory on device (78.84/79.25 GiB) on startup. Desired GPU memory utilization is (0.92, 72.91 GiB). Actual usage is 15.26 GiB for consumed memory (weights + non-torch), 1.44 GiB for peak activation, and 0.23 GiB for CUDAGraph memory.

(EngineCore pid=326) INFO 10-06 17:00:33 [gpu_worker.py:692] Available KV cache memory: 56.21 GiB


Analysis:


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
KV cache per token = (56.21*1024*1024)/460,496 = 128 tokens/KiB

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