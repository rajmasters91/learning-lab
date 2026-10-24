1. Why does INT4 speed up decode but not necessarily prefill? 
The format the weights are stored in and the format the GPU computes in can differ. INT4 weights are usually unpacked to 16-bit for the maths. That's why INT4 shrinks memory and speeds up decode but doesn't necessarily speed up prefill.  ## verify this in week5


2. For gpuaas enabled nodes, nvidia-smi shows persistence-M as off. Why ?
because the nvidia-persistenced service runs with "--no-persistence-mode"


3. first three or four chunks in every run arrive ~6 ms apart, then it settles to 11.5 ms. The very first chunk is a role-only header with no token, but that doesn’t explain all of them. Why is this ? ## Session3


4. What sets the 34 ms TPOT plateau at inf: KV bandwidth or compute? How would I tell them apart?  ## session 4

Both contribute, roughly equally (memory ~16 ms, compute ~10 ms, overhead the rest); 
How to observe:
- shorter contexts reduces KV (VRAM/memory) reads without reducing compute.
- smaller batch (rate1,4) reduces compute faster than memory.


TPOT plateau is because of batching-limit. see [dmon: what the hardware did](weeks/01-baseline/README.md)


5. Why is TTFT at rate 1 (~71 ms) higher than at rate 4 (~41 ms)?   ## Session 4, open

Status: reproducible, server-side, cause unknown.

Evidence:
- Rate 1 TTFT p50 = 70.8 / 70.9 / 71.3 ms across three runs on a clean server (seeds 42, 43, 44).
  Rate 4 TTFT p50 = 40.5 / 41 ms. Rate 16 = 53 ms. So TTFT falls from 1 → 4, then rises again.
- Not queueing: queue_times in the JSON are ~0.1 ms at every rate.
- Not prefix caching: different seeds, random prompts, /v1/completions (no template);
  cache hit rate ≈ 0.
- Not warm-up: the run after a 200-request warm-up gave the same 71 ms as the warm-up itself.
- Not preceding heavy load: a rate-1 run after the rate-64 and inf runs still gave 68 ms.
- Not CPU frequency scaling: governor set to performance → 67.9 ms (schedutil: ~71). Within noise.
- Not the client or network: the server's own /metrics TTFT histogram puts ~780 requests in
  the 60–80 ms bucket, matching the 800 rate-1 requests. The server sees 71 ms too.
- Server config identical between the two sessions (same image, args, KV cache size).

Remaining suspects (for Week 3/4):
- How the scheduler handles a lone prefill under chunked prefill
  (max_num_batched_tokens, long_prefill_token_threshold).
- CUDA-graph coverage: FULL graphs captured for decode batch sizes, PIECEWISE otherwise;
  a prefill arriving with little else running may take the slower path.
- Something that keeps the engine "busy enough" at rate 4 to avoid a per-request setup cost.

Test to try: sweep rates 0.5, 1, 2, 3, 4 to see the shape of the dip; compare a rate-1 run with chunked prefill disabled.

Note: one rate-1 run on 2026-10-11 measured 36 ms TTFT (p99 49, ITL p99 16). It has not been reproduced in five attempts across two server instances and two CPU governors. Logged as an [unexplained outlier](weeks/01-baseline/results/a100-80g/session4_old/rate1_redo); not used as data.