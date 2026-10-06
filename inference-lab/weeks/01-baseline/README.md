1. how much GPU memory will nvidia-smi show in use once the vLLM server is idle, and why?
    It will show 16GB (8B*2) in use since the model is loaded and decoded in VRAM.

