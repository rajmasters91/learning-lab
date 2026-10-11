1. Why does INT4 speed up decode but not necessarily prefill? 
The format the weights are stored in and the format the GPU computes in can differ. INT4 weights are usually unpacked to 16-bit for the maths. That's why INT4 shrinks memory and speeds up decode but doesn't necessarily speed up prefill.

2. For gpuaas enabled nodes, nvidia-smi shows persistence-M as off. Why ?
because the nvidia-persistenced service runs with "--no-persistence-mode"


## Session3

first three or four chunks in every run arrive ~6 ms apart, then it settles to 11.5 ms. The very first chunk is a role-only header with no token, but that doesn’t explain all of them. Why is this ?