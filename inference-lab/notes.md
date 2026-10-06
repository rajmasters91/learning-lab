## Concepts: 

Session1:

What has to happen to generate one token? The model weight has to be read from GPU memory, all of it, every time.
How big is that something for an 8B model in BF16, and why? 16GB
How fast does reading it from VRAM take at 1.94 TB/s? 8ms/token or ~120 token/second.
Why doesn't compute matter here? Roughly how long does the maths take versus the memory read?


bandwidth calculation:

Ex: for meta-llama/Llama-3.1-8B-Instruct model, the bandwidth or token/second calculation is below: 
1. 8B => 8 Billion parameters.
Check config.json file in the hugging face page of the model to identify the model's precision/datatype (dtype) which is "torch_dtype": "bfloat16". Hence BF16 is used for this. Note: this is also found in vLLM startup log (dtype=torch.bfloat16).
Precisions can be BF16, F32, INT8, INT4,etc. The number mentioned is the bits. so BF16 has 16 bits => 2 bytes.
VRAM required to load the model on GPU = 8*2 = 16GB.

A100 card reads at  1.94 TB/s. So, 16 GB divided by 1.94 TB/s or 16 GB / 1940 GB/s = 0.008 or 8ms. 
To generate one token using this model, A100 card takes 8.2 millisecond.
Hence, For 1 second (1/0.0082), ~120 tokens are produced.

Note: Quantization => converting a model from higher precision (ex:F32 or BF16) to a lower precision (INT8, INT4). It increases decode speed but reduces accuracy.
ex: 8B model requires 16GB VRAM when BF16 dtype is used (8x2) but same model in INT4 require only 4GB VRAM (8x0.5)



Why 1.94 TB/s ? i.e., VRAM data transfer speed.

```$ nvidia-smi -i 1 -q -d SUPPORTED_CLOCKS``` -->  Memory clock = 1512 MHz
NVIDIA's pdf "A100 80GB PCIe datasheet" ->   'memory bus width' = 5120-bit and bf16_tflops=312.
HBM class of memory used in GPU (like DDR RAM of server), moves data on both edges of each clock tick (double data rate), so 3,024 million transfers per second per pin. 

Thus calculation is,
1512 MHz
× 2: HBM (1512*2 = 3,024 million transfers per second per pin)
× 5120 bits: "memory bus width". 5,120 bits move in parallel on every transfer.
÷ 8: bits to bytes.
= 1.94TB/s

At 1.94 TB/s, the throughput of ~120 tokens/s is still "memory-bound" and NOT "compute-bound". Why ?? 

- To generate 1 token = 2 arithmetic operations per parameter (a multiply and an add). So:
8 billion × 2 = 16 billion operations per token
The A100 does ~312 trillion BF16 operations per second (bf16_tflops: 312)
16 billion ÷ 312 trillion ≈ 0.05 ms

compute takes 0.05 ms
memory read takes 8.2 ms. 

=> GPU spends about 99% of each step waiting for weights to arrive. That's what memory-bound means: the bottleneck is bandwidth, not calculation.

Why batching works? 

If 8 users decode at once, the model weight is still read only once per step (not 8 times) and generates 8 tokens (1/user).  Here, memory read stays same at 8.2 ms and Compute raises from 0.05ms to 0.4ms (0.05*8). Still compute time is lower than memory read time while throughput goes 8x (8 tokens). 
Thus more tokens are generated for the same amount of time in case of batch jobs.




Xid 79: "GPU has fallen off the bus" => NVIDIA driver has lost communication with the GPU over the PCIe (Peripheral Component Interconnect Express) bus. OS cannot detect or interact with GPU.
Check: dmesg -T | grep -i xid
Causes: Physical connections (reseat GPU, check power cables), H/w failure, overheating (ex: in dg05, GPU0's power was capped @ 200W to avoid xid79 due to overheating) 



NVIDIA GPU Persistence Mode:
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


Question:
For gpuaas enabled nodes, nvidia-smi shows persistence-M as off. Why ?




Tuning Power and GPU Clocks: Need to create a separate systemd service to make these persistent.
 
Ex: To set max power and clocks refer ExecStart in below systemd config file. This persists across reboots.

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


Note:    
-i,   --id=                 Target a specific GPU or Unit.    
-lgc  --lock-gpu-clocks=    Specifies <minGpuClock,maxGpuClock> clocks as a pair (e.g. 1410,1410) => range of desired locked GPU clock speed in MHz.






Session2: 


Card used: A100 80GB
GPU Used: GPU2
Model used:  Llama-3.1-8B-Instruct
vllm image used: vllm/vllm-openai:nightly


docker run --gpus '"device=2"' \
  --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=<redacted>>\
  vllm/vllm-openai:nightly\
  --model meta-llama/Llama-3.1-8B-Instruct


Command Breakdown: 
 --gpus '"device=2"' : assigns GPU with index 2 to the container.
 --ipc=host : shares host's shared memory space with container. Protect vLLM from crashing (due to pytorch shared-memory limits) during inference.
 -p 8000:8000: maps container port 8000 to host port 8000 to expose the webserver.
 -v ~/.cache/huggingface:/root/.cache/huggingface : stores the model in this location in host so that container can reuse the already downloaded model weights and doesnt need to pull it from scratch.
-e HF_TOKEN= : hugging face token passed as environment variable. mandatory for Llama 3.1, as it is a gated model i.e., requires licence acceptance from the model page.
vllm/vllm-openai:nightly : pulls openai vLLM docker image with tag "nightly"
--model meta-llama/Llama-3.1-8B-Instruct : passed as vLLM entrypoint. says which HF model repo to load.


VLLM startup log: findings:

The dtype line (confirms BF16): ```dtype=torch.bfloat16```
How much memory the weights took, and how long loading took:  ```Model loading took 15.0 GiB memory and 22.071041 seconds```
How much memory went to KV cache, and the number of tokens it can hold:  ```Available KV cache memory: 56.21 GiB``` and ```GPU KV cache size: 460,496 tokens```
The max_model_len it chose: ```max model len 131072```
The "maximum concurrency" line, if present: ```GPU KV cache size: 460,496 tokens, Maximum concurrency for 131,072 tokens per request: 3.51x``


## Reference

GPU Clock locking:
GPU's core clock (the SM clock) isn't fixed. It increases with workload and drops when the GPU reaches its power limit or gets hot. => same benchmark can run at 1,410 MHz one minute and 1,250 MHz the next, this affects benchmark numbers.

$ nvidia-smi -i 2 -q -d SUPPORTED_CLOCKS        ### lists the allowed values. For A100, SM clocks range from 210 MHz to 1410 MHz. 

Locking => telling the GPU to stay at one SM value: sudo nvidia-smi -i 2 -lgc 1410,1410.

Note: 1512 MHz memory clock on the A100 is fixed and cant be locked. 

Tradeoff: For A100, 1410 MHz is maximum. GPU under high load can reach 300W and hence drops GPU clock below 1410 MHz even when locked. A lower value (say 1200) holds steady but at reduced speed.

How to decide: start with 1410. During your first benchmark, watch nvidia-smi -i 2 -q -d CLOCK,PERFORMANCE and look at the actual SM clock and the throttle reasons. If it holds 1410, keep it. If it dips, lower the lock until it stays put.

Command used:
1. Diagnostics & Monitoring
⚬	nvidia-smi — View current GPU state, Persistence-M status, and active clocks.
⚬	sudo systemctl status nvidia-persistenced — Check if the persistence daemon is running and view its launch arguments.
⚬	sudo journalctl -eu nvidia-persistenced — View the startup logs and error history for the daemon.
⚬	sudo lsof /dev/nvid* — Verify if the daemon is actively holding the GPU character device files open.
⚬	nvidia-smi -q -d SUPPORTED_CLOCKS — List the maximum hardware-supported memory and graphics clocks for your GPUs.
2. Fixing the Persistence Daemon (Update-Resilient Drop-in)
⚬	sudo mkdir -p /etc/systemd/system/nvidia-persistenced.service.d/ — Create the systemd drop-in directory.
⚬	sudo bash -c 'echo -e "[Service]\nExecStart=\nExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose" > /etc/systemd/system/nvidia-persistenced.service.d/override.conf' — Create the override file to strip the --no-persistence-mode flag.
⚬	sudo systemctl daemon-reload — Tell systemd to re-read the configuration and detect the drop-in.
⚬	sudo systemctl restart nvidia-persistenced — Restart the daemon to apply the persistent state.
3. GPU Tuning & P0 Locking (Targeting Specific GPUs)
⚬	sudo nvidia-smi -i 2 -pm 1 — Manually toggle legacy persistence mode on for GPU 2 (if bypassing the daemon).
⚬	sudo nvidia-smi -i 2 -pl 250 — Set a strict 250W power limit for GPU 2.
⚬	sudo nvidia-smi -i 2 -ac 1215,1410 — Lock application memory and graphics clocks to maximum frequencies for GPU 2.
⚬	sudo nvidia-smi -i 2 -lgc 1410,1410 — Set a hard lock on the core GPU clock bounds to prevent thermal downclocking for GPU 2.
4. Making Automation Scripts Executable
⚬	sudo chmod +x /usr/local/bin/gpu-tune.sh — Grant execution permissions to your custom tuning bash script before hooking it into systemd.