## Concepts: 


## Tokens/second for the given model and GPU:

Ex: for meta-llama/Llama-3.1-8B-Instruct model, the bandwidth or token/second calculation is below: 

8B => 8 Billion parameters.

Precision/Datatype (dtype) of the model: 2 sources
1. config.json file in huggingface page of model: "torch_dtype": "bfloat16". => BF16. 
2. vLLM startup log (dtype=torch.bfloat16)
Precisions - BF16, F32, INT8, INT4, etc.  number mentioned is in bits. => BF16 has 16 bits = 2 bytes.

VRAM required to load the model on GPU = 8*2 = 16GB.

A100 card reads at  1.94 TB/s. So, 16 GB divided by 1.94 TB/s or 16 GB divided by 1940 GB/s = 0.0082 or 8.2 ms. 

To generate one token using this model, A100 card takes 8.2 millisecond.
Hence, For 1 second (1/0.0082), ~120 tokens are produced.


## Quantization vs precision:
Quantization => converting a model from higher precision (ex:F32 or BF16) to a lower precision (INT8, INT4). It increases decode speed but can reduces accuracy. 
ex: 8B model requires 16GB VRAM when BF16 dtype is used but same model in INT4 require only 4GB VRAM (8x0.5)

- accuracy reduction due to quantization is measured in week 5. 


## VRAM/Memory Bandwidth:  Why 1.94 TB/s ? i.e., VRAM data transfer speed.

```$ nvidia-smi -i 1 -q -d SUPPORTED_CLOCKS``` -->  Memory clock = 1512 MHz
NVIDIA's pdf "A100 80GB PCIe datasheet" ->   'memory bus width' = 5120-bit and bf16_tflops=312.
HBM class of memory used in GPU (like DDR RAM of server), moves data on both edges of each clock tick (double data rate i.e., 1512*2 = 3,024 million transfers per second per pin). 

Thus calculation is,
1512 MHz
× 2: HBM (1512*2 = 3,024 million transfers per second per pin)
× 5120 bits: "memory bus width". 5,120 bits move in parallel on every transfer.
÷ 8: bits to bytes.
= 1.94TB/s

- HBM = High Bandwidth Memory

## Memory Bound or Compute Bound ? 

At 1.94 TB/s, the throughput of ~120 tokens/s is still "memory-bound" and NOT "compute-bound". Why ?? 

- To generate 1 token = 2 arithmetic operations per parameter (a multiply and an add).
So,
  8 billion × 2 = 16 billion operations per token.
  A100 does ~312 trillion operations per second (bf16_tflops: 312)
  16 billion ÷ 312 trillion ≈ 0.05 ms

compute takes 0.05 ms
memory read takes 8.2 ms. 

=> GPU spends about 99% of each step waiting for weights to arrive. That's what memory-bound means: the bottleneck is bandwidth, not calculation.

## Why batching works? 

If 8 users decode at once, the model weight is still read only once per step (not 8 times) and generates 8 tokens (1/user).  Here, memory read stays same at 8.2 ms and Compute raises from 0.05ms to 0.4ms (0.05*8). Still compute time is lower than memory read time while throughput goes 8x (8 tokens). 
Thus more tokens are generated for the same amount of time in case of batch jobs.


### storage vs work: 
- **Bytes**: each parameter is stored in BF16 = 2 bytes. 8 billion × 2 bytes = 16 GB of memory. => **storage**
- **Operations**: generating one token does ~2 arithmetic operations per parameter (a multiply and an add). 8 billion × 2 = 16 GFLOP of compute per token. => **work**
- Thus, 16 GB (storage, bytes) and 16 GFLOP (work, operations) are different quantities that coincidentally share a number

## Xid 79: 
=> "GPU has fallen off the bus" => NVIDIA driver has lost communication with the GPU over the PCIe (Peripheral Component Interconnect Express) bus. OS cannot detect or interact with GPU.
Check: dmesg -T | grep -i xid
Causes: Physical connections (reseat GPU, check power cables), H/w failure, overheating (ex: in node1, GPU0's power was capped @ 200W to avoid xid79 due to overheating) 
Workaround: 
- cold reset from console or via IPMI

## NVIDIA GPU Persistence Mode:
- Prevents OS kernel from unloading NVIDIA driver even when no processes are actively using the GPUs.
- Why needed : to eliminate driver initialization latency for compute/headless workloads.

Note:
- compute workload: mathematical operations that use GPU only for parallel processing. 
- headless: server without attached monitor or GUI. managed via ssh and do not have display manager like X11

NVIDIA supports two methods for achieving state persistence:
	1.	Legacy Kernel Persistence:  nvidia-smi -pm 1. 
		- instructs kernel-space driver to stay awake.  
		- Persistence-M column in nvidia-smi will show "On" when enabled.
	2.	User-Space Daemon (nvidia-persistenced): 
       	- Holds GPU character device files open. 
		- By maintaining an open file descriptor, it prevents the Linux kernel from unloading the driver state.
      	- Ex: In ubuntu, nvidia-persistenced is configured with --no-persistence-mode flag: ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --no-persistence-mode
		
```
      root@node1:~# sudo systemctl edit nvidia-persistenced
      Successfully installed edited file '/etc/systemd/system/nvidia-persistenced.service.d/override.conf'.
      
      root@node1:~# cat /etc/systemd/system/nvidia-persistenced.service.d/override.conf
      ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose
      
      root@node1:~# sudo systemctl daemon-reload ; sudo systemctl restart nvidia-persistenced
```

Note: Use drop-in file. Do not edit /usr/lib/systemd/system/nvidia-persistenced.service directly. package update will overwrite it.

Validation: 
      1. $ nvidia-smi => "Persistence-M: On" for all GPUs.
      2. lsof /dev/nvidia* => must list the open GPU character device files (/dev/nvidiactl, /dev/nvidia0, /dev/nvidia1)
	  
	  ```
        	   root@node1:~# lsof /dev/nvidia*
               COMMAND     PID                USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
               nvidia-pe 56684 nvidia-persistenced    2u   CHR 195,255      0t0  997 /dev/nvidiactl
               nvidia-pe 56684 nvidia-persistenced    3u   CHR   195,0      0t0 1001 /dev/nvidia0
               nvidia-pe 56684 nvidia-persistenced    5u   CHR   195,0      0t0 1001 /dev/nvidia0
			   ```

## GPU Clock locking:
GPU's core clock (SM clock) isn't fixed. i.e., increases with workload ; drops when GPU reaches its power limit (W) or gets Hot (C). => same benchmark can run at 1,410 MHz one minute and 1,250 MHz the next, this affects benchmark numbers.

Locking => telling the GPU to stay at one SM value: sudo nvidia-smi -i 2 -lgc 1410,1410.

nvidia-smi -q -d SUPPORTED_CLOCKS  # Lists maximum hardware-supported memory & graphics/SM clocks for GPUs.

Ex: For A100, 
SM clocks range from 210 MHz to 1410 MHz -> can be locked
memory clock fixed at 1512 MHz -> cant be locked. 

Ex: nvidia-smi -i 2 -lgc 1410,1410 — lock min & max gpu clock @1410 MHz to prevent thermal downclocking for GPU 2.

Tradeoff: For A100, 1410 MHz is maximum. High load --> GPU Power reach 300W --> drops GPU clock below 1410 MHz (even when locked). A lower value (say 1200) holds steady but at reduced speed.

How to decide: start with 1410. During your first benchmark, watch nvidia-smi -i 2 -q -d CLOCK,PERFORMANCE and look at the actual SM clock and the throttle reasons. If it holds 1410, keep it. If it dips, lower the lock until it stays put.


## Tuning Power and GPU Clocks: 

Need to create a separate systemd service to make these persistent.
 
Ex: To set max power and clocks refer ExecStart in below systemd config file. This persists across reboots.

```
root@node1:~# systemctl cat gpu0-thermal.service
# /etc/systemd/system/gpu0-thermal.service
[Unit]
Description= GPU0 thermal mitigation (power cap + clock lock) - Xid 79 workaround
After=nvidia-persistenced.service multi-user.target
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
# GPU0 (PCI 01:00) runs ~30C hotter than peers and throws Xid 79 under load.
# Cap power + clocks to reduce peak heat. || true so a missing/healthy GPU0
# never blocks boot. Remove this unit once GPU0 cooling is serviced.
ExecStart=/bin/sh -c "/usr/bin/nvidia-smi -i 0 -pl 200 || true"   ## set Max power to 200W for GPU0.
ExecStart=/bin/sh -c "/usr/bin/nvidia-smi -i 0 -lgc 210,1200 || true" ## sets minGpuClock,maxGpuClock for GPU0

[Install]
WantedBy=multi-user.target

root@node1:~#
root@node1:~# sudo systemctl daemon-reload
root@node1:~# sudo systemctl restart gpu0-thermal.service
```

Note:    
-i,   --id=                 Target a specific GPU or Unit.    
-lgc  --lock-gpu-clocks=    Specifies <minGpuClock,maxGpuClock> clocks as a pair (e.g. 1410,1410) => range of desired locked GPU clock speed in MHz.


## Units: 
GB (decimal, 10⁹ bytes): datasheets and “16 GB for 8B params”
GiB (binary, 1024³ bytes): vLLM logs
MiB (1024² bytes): nvidia-smi


## Cold vs warm (Page cache) start. 
Run 1: model loading 22.07 s. 
Run 2: 5.42 s, with “Loading weights took 0.87 s”. Same files, same disk. 

What changed between runs? 
Linux page cache (buff/cache column of free -h) -> model is cached and hence run 2 took less time.
BUT,
Compilation stayed at ~15 s both times, so it isn’t cached across container restarts. 


## KV Cache : 

KV cache 
- memory of each computed tokens stored by the model => no recompute of past tokens in subsequent steps of same prompt/request.
- grows with context x concurrency
- 56.21GiB for A100 (128KiB/token)
note: no recompute of past tokens in subsequent prompts →  prefix caching.

### Reference From Gemini

KV Cache (Key-Value Cache) =  memory optimization technique used in LLMs & Transformer-based AI models to speed up text generation process.

Problem It Solves
LLMs generate text "autoregressively" => they predict one word (token) at a time. or 1 "Step" at a time.
To predict the next word or during "Step" 2, the model needs to look back at all the previous words in the conversation to understand the context. 
No cache => model recalculate the mathematical representations of every single previous word for every new word it generates. = Latency + compute time will increase - affects batching.

How KV Cache Works
In a Transformer model, the "Attention" mechanism uses 3 vectors for every token: Queries (Q), Keys (K), and Values (V).
1. Query => what the current token is looking for.
2.  Key => label of past token.
3.  Value =>  content of that past token.
Model calculates Keys & Values for a particular step or token and saves them in KV cache.
When generating a new token (i.e., a For subsequent step), model only computes Query, Key, & Value for that new single token, and compares its Query against the cached Keys & Values of the past.

Pros and Cons
- speeds up text generation (lowers latency) and saves compute.
- Trade-off: Storing these vectors takes up a lot of memory (VRAM). i.e., longer conversation (larger context window) => large KV cache = OOM due to a massive KV cache. --> main bottleneck in hosting LLMs.


## prefix caching:
To reduce TTFT for recurring queries/prompts.
"ON" by default.
If we change first few words for the same prompt, prefix caching will miss => TTFT will be higher.

Due to prefix caching, for long prompts, Run1 (no cache) took 0.904s whereas Run2 took only 0.762s (prefix cache from Run1) 

check below log check "Long prompt evaluated with time command:", 
[View Session3 Baseline logs](01-baseline/results/a100-80g/session3/Session3_stream_run-gpu2-2026-10-11.txt)


## TTFT vs TPOT: Time To First Token vs Time Per Output Token

LLM generation = 2 phases
1. Prefill: Reading prompt
2. Decode: generating response

LLM Metrics: 
- TTFT: Prefill Phase. 
     - Time b/w user submitting the prompt & First token generation.
     - Compute-bound.
     - KV Cache is built
     - measures latency

- TPOT (Inter-token latency/ITL): Decode phase
     - average time to generate each subsequent token after first one.
     - memory-bandwidth bound.
     - measures per token latency.

## chat template overhead: ex: "/v1/chat/completions" endpoint 

What it is: chat endpoint doesn’t send user prompt to model as-is. vLLM server wraps user prompt in model’s chat template. 
For Llama 3.1, wrapper is 35 tokens, and includes a hardcoded “Today Date: 26 Jul 2024” line.

chat template = special tokens that mark start of text, who is speaking (system, user, assistant), and where each turn ends, plus any default system text

Where I saw it: /tokenize returned 40 tokens for a 5-token sentence. IDs above 128000 (128000, 128006, 128007, 128009) are the structural special tokens; 9125/882/78191 are “system”/”user”/”assistant”.

Why it matters:

Short prompts are mostly template. 
35 of 40 tokens were overhead, so for a one-line question prefill is dominated by the wrapper, not the user.
Wrapper is identical for every request --> prefix caching =>  Those 35 tokens are computed once and served from cache afterwards.

/v1/completions (the raw endpoint) adds no template; used to control the exact tokens, as a benchmark tool does.



## End-to-End Latency 

End-to-End Latency = TTFT + [(Total Output Tokens -1)*TPOT]

Tradeoff in optimizing vLLM: 
- max throughput via batching => increased TTFT for individual users.
- Optimising for low TTFT alone (small batches) => lowers throughput (total output tokens)

## Steps, continuous batching, chunked prefill (the 16 vs 46 ms argument)

```
[ Incoming User Prompts ] ──┐
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 1. THE SCHEDULER                                       │
│ Batches decoding requests and chunks of new prompts    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. THE MEMORY FETCH (The Physical Speed Limit)         │
│ ├─ Stream all Model Weights (e.g., 15 GB) from HBM     │
│ └─ Fetch the KV Cache (past context) for the batch     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. THE COMPUTE (Matrix Multiplication)                 │
│ Multiply the 15 GB of weights against the batch data.  │
│ (Idle at batch size 1; fully saturated at batch 200)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. EMIT & UPDATE                                       │
│ ├─ Output exactly 1 new token per active request       │
│ ├─ Save these new tokens back into the KV Cache        │
│ └─ Stream tokens over the network to users             │
└───────────────────────────┬────────────────────────────┘
                            │
                            └─── ( Loop Starts Over ) ───>

```
Constants: 
- Model weights (15GB) is read (~8.2ms) for every step irrespective of number of requests (1 or 200). => batching increases throughput.
- See [session3 stream - long prompt](weeks/01-baseline/results/a100-80g/session3/Session3_stream_run-gpu2-2026-10-11.txt) 
     - Each "data:" line is output of 1 step => 11.3ms gap between each step (token gap).
     - Token gap of 16ms => that step was doing extra work (someone's prefill chunk i.e., "chunked prefill")
     - Hence, for a request/prompt,
          - TTFT = how many steps for prefill + how long each prefill step took
          - TPOT = how long each later step took.
     - Session4 load test:
          rate 1 => steps carry 1-2 tokens and takes 11ms
          inf => steps carry 200 tokens, take 34ms and produces ~200 tokens each step. (200/0.034 = ~5900 tokens - overhead)

## Request rate vs concurrency, including Poisson bunching (average ≈ rate × duration; peaks several× higher)

`max_concurrent_requests: 8–9` at Rate 1: 
Rate 1 = 1 request/s ; Each request lasts ~1.5 s => *average* number of requests in flight = 1–2.
But, peak was 8–9. Why? 
     Poisson arrival pattern:
     - requests bunch up randomly, the way real users do.
     - Explains why even rate 1 has a p99 ITL above the median: during a bunch, several prefills land on the same steps as your decode.

Take it in three parts.

1. Average concurrency: => average number of requests on the GPU at a point in time.

Rate 1 => 1 request/s (Rate16 => 16req/s; Rate inf => infinite request/s)
Each request lives in GPU for ~1.5 s  (36 ms TTFT + 128 tokens × 11.3 ms ≈ 1.5 s) 
Thus, *average* number of requests in flight = 1–2 => Roughly one or two requests are in progress at any moment. => average load.

2. Peak concurrency: 8–9. Why so much higher than 1–2?
The quering tool uses "Poisson arrival pattern" to mimic real world traffic. 
It randomly sends bunch of requests (clusters) - biggest recorded cluster was 8-9.

3. Why this shows up in p99 ITL.

At Rate1, requests are decoded @ 1 token per step (~11.2 ms each). 
Clusters (Poisson) arrive, a bunch of 3 new requests with each needing 512 token prefill. 
vLLM server adds those prefill chunks into same steps as decode (chunked prefill) => A Step now had 1 decode token + prefill chunck => longer ITL ex: 16ms instead of 11ms.

See [session 3 stream](weeks/01-baseline/results/a100-80g/session3/Session3_stream_run-gpu2-2026-10-11.txt)

Most of the 127 token gaps are undisturbed (median 11.3 ms). 
Those coincided with clusters are slow.
Slowest 1% of them is the p99 (16.4 ms). => Even at light average load, the tail feels the bunching.

At rate 4 the clusters are bigger and more frequent => gap between median ITL and p99 ITL will widen. 
Predict by how much before you run it.


## Mean, median, std, p99, and what each is for

[Take run 1](weeks/01-baseline/results/a100-80g/session4_old/rate1/vllm-1.0qps-Llama-3.1-8B-Instruct-20261011-200342.json): 200 requests, 200 TTFT values.

Four ways to summarising this 200 TTFT measurements:


1. Mean (average): 
- add all 200 TTFT and divide by 200 → 36.6 ms. 
- Issue: Sensitive to outliers.
         i.e., one freak request at 2,000 ms will change the mean to 46 ms even if 199 users saw nothing unusual. 

2. Median (p50 - 50th percentile): 
- sort the 200 values from lowest to highest. Median => middle one → 36.2 ms. 
- Half the requests were faster, half slower. 
- one outlier (2,000 ms) request do not skew this value. 
- This is “What a typical user saw.”
- Lesson:  mean ≈ median, as here (36.6 vs 36.2) =>  distribution is symmetric with no big outliers. Even in the slow warm-up run (68.9 vs 70.9), mean ≈ median => slowness was everywhere, not a few bad requests.

3. Standard deviation (std): 
- How spread out the values are around the mean.
- Small std => measurements are tightly clustered; large std => token gap is wide/
- Lesson: For benchmarking, std is the repeatability signal: if two runs have the same median but one has triple the std, something is unstable.
  - Ex:
	- Run 1: std 4.8 ms, so most TTFTs fell within roughly 32–41 ms.
 	- Warm-up: std 11.5 ms, more than twice as scattered. 

4. p99 (99th percentile): 
- sort the 200 values; take the one at 99th percentile (the 198th). 
- Only 1% of requests were slower → 49 ms in run 1. 
- “What the unluckiest 1 in 100 users saw.” This is the tail. 
- Used by ops teams in SLOs, because a service that’s fast on average but terrible 1% of the time still has angry users at scale.

How they work together: 
- median = typical case, 
- p99 = tail, 
- std = how trustworthy the run is, 
- mean is mostly a cross-check against median. 

Session4_old observation: 
- warm-up effect:
     - showed up strongly in std and p99 (the tail)
     - only mild in the median (the typical case).


## Offered rate vs achieved rate = saturation

offered rate = request rate we asked for. ex: At rate 32, we asked for 32 requests to be sent to vLLM. 
achieved rate = request_throughput; 

achieved < offered => vLLM server can’t keep up, requests queue = saturation. 

Evidence: 

At [Rate 32](weeks/01-baseline/results/a100-80g/session4_old/rate32/vllm-32.0qps-Llama-3.1-8B-Instruct-20261011-201148.json):

"request_rate": 32.0, "request_throughput": 23.349323010233036, 

At [Rate 64](weeks/01-baseline/results/a100-80g/session4_old/rate64/vllm-64.0qps-Llama-3.1-8B-Instruct-20261011-201438.json):

"request_rate": 64.0, "request_throughput": 30.137445138057984, 


## FALSIFIED HYPOTHESIS 

In session4_old results, Warm-up : TTFT 71 → 36 ms; decode (TPOT) unaffected; why? 

Assumption:  comparing warm-up and run1, TTFT reduces in run1 due to prefill cache from warm-up run. --> WRONG

warm-up used seed 42 & run 1 seed 43 => so the prompts were different
random /v1/completions prompts share no prefix anyway. 

HYPOTHESIS: Warm-up is a different mechanism - first-time costs inside the engine (kernels loaded, CUDA graph paths exercised, memory allocator settled) that are paid once after a restart. They sit on the first-token path =>  higher TTFT for warmup run.
Hence, for run1 TTFT and the tails improved while steady (~11.2ms. ) decode didn’t.

warm-up costs are paid the first time (TTFT) the engine meets a new kind of step; decode(TPOT) steps repeat the same few kinds, so they warm up within seconds, while prefill meets new variations throughout the first run.

TEST: In session4, did a clean restart of vLLM and did three rate-1 runs with different seeds; no warm-up effect found. 
STATUS: no warm-up effect visible at this scale with this engine. 



**Prefix caching is about data**
**Warm-up is about code**


## Clock lock vs power cap (the lock is a ceiling)

clock lock at max (1410 MHz) is a ceiling but it will come down if the power cap of 300W is hit. 
We cannot guarantee a fixed clock lock even if we set it using a systemd service.  

[dmon evidence](weeks/01-baseline/results/a100-80g/session4_old/rate_inf/run1/nvidia-smi_dmon_ran_on_second_terminal.txt)

```
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk     fb   bar1   ccpm 
# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz     MB     MB     MB 
    2     60     35     46      0      0      0      0      0      0   1512   1410  74569  74569      0 
    2     60     38     48      0      0      0      0      0      0   1512   1410  74569  74569      0 
    2    302     48     53    100     63      0      0      0      0   1512   1245  74569  74569      0 
    2    296     50     54    100     64      0      0      0      0   1512   1245  74569  74569      0 
    2    294     51     56    100     64      0      0      0      0   1512   1245  74569  74569      0 
    2    296     52     57    100     65      0      0      0      0   1512   1245  74569  74569      0 
    2     63     42     52     21     15      0      0      0      0   1512   1410  74569  74569      0 
    2     61     38     50      0      0      0      0      0      0   1512   1410  74569  74569      0
``` 

## FLOP vs FLOPS:
  
FLOP is an amount of work; FLOPS (per second) is a rate.

For A100 card, at 312 TFLOPS (i.e.,TFLOP/s), 2000 token prompt takes ~104 ms (33/312) for prefill compute.

T=Tera (10^12)

## Why batching works needs its second half: the limit. KV reads and compute at high concurrency; the 34 ms arithmetic.

See 
[session4_old rate16](weeks/01-baseline/results/a100-80g/session4_old/rate16/run1/vllm-16.0qps-Llama-3.1-8B-Instruct-20261011-191000.json)
[session4_old rate64](weeks/01-baseline/results/a100-80g/session4_old/rate64/vllm-64.0qps-Llama-3.1-8B-Instruct-20261011-201438.json)

First half: (Rate 1, 4, 8)
At a batch of 8 requests, model weights are read once (8.2 ms) & compute rises only 0.05 → 0.4 ms, so 8 tokens come out for roughly the price of 1 (11.2ms TPOT)
Assumption: batching is free forever i.e., ulimited requests = unlimited tokens.
Reality:  TPOT went 11.3(rate4) → 15(rate16) → 32 ms(rate64). 

Second half: (Rate 16, 32, 64, inf) => *why it stops being free*.

Two things grow with the batch size:

**1. KV cache reads.** 
During Every step, each request's attention has to read that request's KV cache.

One request with ~580 tokens (512 prompt + about 64 output on average mid-run) has 580 × 128 KiB ≈ 74 MB of KV cache. => Trivial next to 15 GB of weights. 

But 200 requests × 74 MB ≈ **15 GB** = same size as the weights.=> Memory traffic per step has doubled => memory read time doubles from ~8ms to ~16 ms.

**2. Compute.** 
One token costs 16 GFLOP; 200 tokens cost 200 × 16 GFLOP = 3.2 TFLOP. At 312 TFLOP/s that's ~**10 ms**. 
At batch 1 compute time(0.05 ms) was less than memory read time (11ms); 
At batch 200 = 16ms memory(weights+KV) + 10ms compute (compute) + overhead and imperfect overlap = ~34ms

| Offered rate | Concurrent requests | KV read per step | Compute per step | Predicted TPOT | Measured |
| --- | --- | --- | --- | --- | --- |
| 1 | ~1–2 | 0.1 GB | 0.05 ms | ~11 ms (memory floor + overhead) | 11.3 |
| 4 | ~6 | 0.4 GB | 0.3 ms | ~11.5 ms | 11.7 |
| 16 | ~25 | 1.8 GB | 1.3 ms | ~13–14 ms | 15.0 |
| inf | ~200 | ~15 GB | ~10 ms | ~30+ ms | 34.0 |

Lesson:
batching is free while the model weights dominate memory traffic & compute time is less than memory read time. 
It stops being free when, whichever below comes first.
1. Batch's total KV cache >= size of the model weights, (or) 
2. compute time = memory read time.  

On A100 card with this model, batching stops free around 25–50 concurrent requests, which is exactly where the knee showed up.

rule of thumb: *batching stops being free when batch KV ≈ weights or compute ≈ memory time.* 

What sets the 34 ms plateau?
both, roughly equally, which is why neither halving compute nor halving KV traffic alone would get you back to 11 ms.



## What is a knee ?

knee = point on a load curve where adding more load stops being cheap and starts being expensive. 

The name comes from the shape: if you plot throughput against load, the line rises steeply, then bends over and flattens, like a leg bending at the knee.

Below the knee: more requests → much more throughput, almost no extra latency. Each new user is nearly free.

Above the knee: more requests → little extra throughput, much more latency. Each new user makes everyone wait.

In your data:

Offered rate	Output tok/s	          TPOT p50
1 → 4	     127 → 497 (4×)	          11.3 → 11.7 (flat)
4 → 16	     497 → 1,801 (3.6×)	     11.7 → 15.0 (+30%)
16 → 32	     1,801 → 2,989 (1.7×)	15.0 → 24.5 (+60%) # knee
32 → 64	     2,989 → 3,858 (1.3×)	24.5 → 32.8 (+35%)

Up to 16, each doubling of load buys nearly a doubling of throughput for small latency cost. Between 16 and 32 the ratio flips: throughput gains shrink, latency jumps. That bend is the knee, and it’s where the achieved rate also stopped keeping up with the offered rate.

Why ops people care: knee is where you want to run a server. 
- Below knee => you’re wasting hardware (GPU could serve more users for free). 
- Above knee => you’re wasting users’ time (p99 latency blows past any SLO). 

Autoscaling, admission control and capacity planning are all about keeping the system just below its knee. 
Week 14’s SLO work is essentially “find the knee, then set the limit a bit under it”.



## sm% & mem% in dmon logs: 
Ex: [session_4_rate1_dmon](weeks/01-baseline/results/a100-80g/session4/rate1/run1/dmon.txt)

Both are percentage of time, NOT percentage of capacity utilization. 

**sm%** (same as GPU-Util% in nvidia-smi): 
- fraction of the last second during which at least one kernel was running on GPU. 
- Ex: 
     - sm% of 94% => GPU was doing something 94% of the last second.
     - Do not say anything about how much % of GPU was being used in that 94% of time. 
     - ie., One tiny kernel keeping one SM busy continuously shows sm% as 100%. 
     - At session4's rate 1 runs, sm% was 94% because a decode step is almost always in flight, even though each step uses only a fraction of the gpu compute.

**mem%**: 
- fraction of the last second during which the memory controller was busy reading or writing HBM. 
- => memory bandwidth utilisation NOT memory capacity (capacity is shown in the "fb" column in dmon logs).
- Ex:
     - At session4's rate 1 runs, mem% of 80% means the memory bus was moving data four seconds out of five => memory-bound decode.

Interpretation: 
- sm% high, mem% high => GPU is busy & memory bound. Rates 1–4.
- sm% high, mem% falling => GPU is still busy but less of its time is spent waiting on memory, so compute bound. Ex: Rates 32–inf: 100% / 60%. batching-limit.
- sm% low => GPU is idle part of the time; bottleneck is elsewhere (client, CPU, queueing).

Limits: these are coarse 1 Hz counters. They can’t tell you how many SMs were active or how close to peak bandwidth you were; for that you need DCGM’s finer counters or Nsight (week3 profiling).



## CPU frequency governors

Linux CPU governor decides how fast each core runs. 
- `schedutil`(default) scales frequency up and down with load, saving power when idle;
- `performance` pins every core at its maximum clock.

How it affects inference: 

TTFT includes CPU-side work (tokenising, scheduling,
launching the prefill kernels). 
At low request rates the cores idle between bursts and can ramp down, so the next request starts on a slow clock. Symptom: "Latency improves under load"

Tested 2026-10-18 on node1: rate 1, schedutil ≈ 71 ms TTFT vs performance 67.9 ms.
Within noise: not the cause of the rate-1 TTFT puzzle on this node. Restored to schedutil.

Rule: check the governor on every new node before benchmarking, and record it in nodes.

