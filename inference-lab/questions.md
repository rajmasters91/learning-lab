1. The format the weights are stored in and the format the GPU computes in can differ. INT4 weights are usually unpacked to 16-bit for the maths. That's why INT4 shrinks memory and speeds up decode but doesn't necessarily speed up prefill.

2. For gpuaas enabled nodes, nvidia-smi shows persistence-M as off. Why ?
because the nvidia-persistenced service runs with "--no-persistence-mode"


