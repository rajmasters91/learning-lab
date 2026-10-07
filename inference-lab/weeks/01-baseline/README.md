1. how much GPU memory will nvidia-smi show in use once the vLLM server is idle, and why?


My assumption:  It will show 16GB (8B*2) in use since the model is loaded in VRAM. >>> WRONG

Result: see nvidia-smi below. 
vLLM grabs almost all VRAM front and pre-allocates KV cache, whether or not any request has arrived. 
Memory in use => budget vLLM was given, NOT how big the model is.


root@dg05:~#
Every 1.0s: nvidia-smi -i 2                                             dg05: Tue Oct  6 17:07:38 2026

Tue Oct  6 17:07:38 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.91.07              Driver Version: 595.91.07      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   2  NVIDIA A100 80GB PCIe          On  |   00000000:81:00.0 Off |                    0 |
| N/A   39C    P0             61W /  300W |   74143MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    2   N/A  N/A          175134      C   VLLM::EngineCore                      74134MiB |
+-----------------------------------------------------------------------------------------+


