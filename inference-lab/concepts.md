## Concepts: 


What has to happen to generate one token? The model weight has to be read from GPU memory, all of it, every time.
How big is that something for an 8B model in BF16, and why? 16GB
How fast does reading it from VRAM take at 1.94 TB/s? 8ms/token or ~120 token/second.
Why doesn't compute matter here? Roughly how long does the maths take versus the memory read?


## Tokens/second for the given model and GPU:

Ex: for meta-llama/Llama-3.1-8B-Instruct model, the bandwidth or token/second calculation is below: 

8B => 8 Billion parameters.

Check config.json file in the hugging face page of the model to identify the model's precision/datatype (dtype) which is "torch_dtype": "bfloat16". Hence BF16 is used for this. Note: this is also found in vLLM startup log (dtype=torch.bfloat16).
Precisions can be BF16, F32, INT8, INT4,etc. The number mentioned is the bits. so BF16 has 16 bits => 2 bytes.
VRAM required to load the model on GPU = 8*2 = 16GB.

A100 card reads at  1.94 TB/s. So, 16 GB divided by 1.94 TB/s or 16 GB divided by 1940 GB/s = 0.008 or 8ms. 
To generate one token using this model, A100 card takes 8.2 millisecond.
Hence, For 1 second (1/0.0082), ~120 tokens are produced.


## Quantization vs precision:
Quantization => converting a model from higher precision (ex:F32 or BF16) to a lower precision (INT8, INT4). It increases decode speed but reduces accuracy.
ex: 8B model requires 16GB VRAM when BF16 dtype is used (8x2) but same model in INT4 require only 4GB VRAM (8x0.5)



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


## Memory Bound or Compute Bound ? 

At 1.94 TB/s, the throughput of ~120 tokens/s is still "memory-bound" and NOT "compute-bound". Why ?? 

- To generate 1 token = 2 arithmetic operations per parameter (a multiply and an add). So:
8 billion × 2 = 16 billion operations per token
The A100 does ~312 trillion operations per second (bf16_tflops: 312)
16 billion ÷ 312 trillion ≈ 0.05 ms

compute takes 0.05 ms
memory read takes 8.2 ms. 

=> GPU spends about 99% of each step waiting for weights to arrive. That's what memory-bound means: the bottleneck is bandwidth, not calculation.

## Why batching works? 

If 8 users decode at once, the model weight is still read only once per step (not 8 times) and generates 8 tokens (1/user).  Here, memory read stays same at 8.2 ms and Compute raises from 0.05ms to 0.4ms (0.05*8). Still compute time is lower than memory read time while throughput goes 8x (8 tokens). 
Thus more tokens are generated for the same amount of time in case of batch jobs.




## Xid 79: 
=> "GPU has fallen off the bus" => NVIDIA driver has lost communication with the GPU over the PCIe (Peripheral Component Interconnect Express) bus. OS cannot detect or interact with GPU.
Check: dmesg -T | grep -i xid
Causes: Physical connections (reseat GPU, check power cables), H/w failure, overheating (ex: in dg05, GPU0's power was capped @ 200W to avoid xid79 due to overheating) 



## NVIDIA GPU Persistence Mode:
- Prevents OS kernel from unloading NVIDIA driver even when no processes are actively using the GPUs.
- Why needed : It eliminates driver initialization latency for compute/headless workloads.

Note:
- compute workload: mathematical operations that use GPU only for parallel processing. 
- headless: server without attached monitor or GUI. managed via ssh and do not have display manager like X11

NVIDIA supports two methods for achieving state persistence:

	1.	Legacy Kernel Persistence: 
      Managed via nvidia-smi -pm 1. This instructs the kernel-space driver to stay awake. The Persistence-M column in nvidia-smi will show "On" when enabled.

	2.	User-Space Daemon (nvidia-persistenced): 
      It holds the GPU character device files open. By maintaining an open file descriptor, it prevents the Linux kernel from unloading the driver state.
      In ubuntu, nvidia-persistenced is configured with --no-persistence-mode flag: ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --no-persistence-mode

      Ex:
      root@dg05:~# sudo systemctl edit nvidia-persistenced
      Successfully installed edited file '/etc/systemd/system/nvidia-persistenced.service.d/override.conf'.
      
      root@dg05:~# cat /etc/systemd/system/nvidia-persistenced.service.d/override.conf
      ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose
      
      root@dg05:~# sudo systemctl daemon-reload ; sudo systemctl restart nvidia-persistenced

      Note: Do not edit /usr/lib/systemd/system/nvidia-persistenced.service directly. package update will overwrite it.

	   Validation: 
      1. $ nvidia-smi => "Persistence-M: On" for all GPUs.
      2. lsof /dev/nvidia* => must list the open GPU character device files (/dev/nvidiactl, /dev/nvidia0, /dev/nvidia1)

         Ex: root@dg05:~# lsof /dev/nvidia*
               COMMAND     PID                USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
               nvidia-pe 56684 nvidia-persistenced    2u   CHR 195,255      0t0  997 /dev/nvidiactl
               nvidia-pe 56684 nvidia-persistenced    3u   CHR   195,0      0t0 1001 /dev/nvidia0
               nvidia-pe 56684 nvidia-persistenced    5u   CHR   195,0      0t0 1001 /dev/nvidia0


## GPU Clock locking:
GPU's core clock (the SM clock) isn't fixed. It increases with workload and drops when the GPU reaches its power limit or gets hot. => same benchmark can run at 1,410 MHz one minute and 1,250 MHz the next, this affects benchmark numbers.

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
root@dg05:~# systemctl cat haishare-gpu0-thermal.service
# /etc/systemd/system/haishare-gpu0-thermal.service
[Unit]
Description=HAISHARE GPU0 thermal mitigation (power cap + clock lock) - Xid 79 workaround
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

root@dg05:~#
root@dg05:~# sudo systemctl daemon-reload
root@dg05:~# sudo systemctl restart nvidia-persistenced
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




## KV Cache : From Gemini

KV Cache (Key-Value Cache) =  memory optimization technique used in LLMs & Transformer-based AI models to speed up text generation process.

Problem It Solves
LLMs generate text "autoregressively" => they predict one word (token) at a time. 
To predict the next word, the model needs to look back at all the previous words in the conversation to understand the context. Without a cache, the model would have to recalculate the mathematical representations of every single previous word for every new word it generates. This is incredibly slow and computationally expensive.

How KV Cache Works
In a Transformer model, the "Attention" mechanism uses 3 vectors for every token: Queries (Q), Keys (K), and Values (V).
⚬	Query => what the current token is looking for.
⚬	Key => label describing a past token.
⚬	Value => actual content of that past token.
Instead of recalculating the Keys and Values for past tokens every step, the model calculates them once and saves them in the KV Cache. When generating a new token, the model only computes the Query, Key, and Value for that new single token, and compares its Query against the cached Keys and Values of the past.

Pros and Cons
⚬	speeds up text generation (lowers latency) and saves compute.
⚬	Trade-off: Storing these vectors takes up a lot of memory (VRAM). As a conversation gets longer (larger context window), the KV cache grows rapidly. OOM due to a massive KV cache is main bottlenecks in hosting LLMs today.




## prefix caching:
prefix caching is used to reduce TTFT for recurring queries.
Prefix caching needs to be true in config.
If we change first few words for the same prompt, prefix caching will miss and TOFT will be higher.

## TTFT vs TPOT: Time To First Token vs Time Per Output Token

## chat template overhead: 
