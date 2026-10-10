# Inference Lab

Hands-on, one-variable-per-week benchmarks of LLM inference serving,
run on whatever GPU I have that week. Plan and reading track:
[Inference Lab: 16-Week Hardware-Agile Benchmark Series](https://claude.ai/artifact/UgAHqPhyWvku32yNkgWujj)

Each week changes exactly one variable, predicts the result first,
measures it, and writes up the bottleneck and the tradeoff.

## Layout

- `concepts.md` — concepts in my own words
- `cheatsheet.md` — commands I actually use
- `questions.md` — open questions, tagged with the week they belong to
- `hardware/` — one profile per GPU model
- `nodes/` — per-machine facts (topology, quirks)
- `setup/` — Ansible playbook to prepare a new node
- `weeks/NN-name/` — one folder per experiment: README, config, logs

## Weeks

| Week | Experiment | Card | Variable | Headline result |
| --- | --- | --- | --- | --- |
| 1 | [Baseline](weeks/01-baseline/) | A100 80GB PCIe (GPU 2, dg05) | none | vLLM idles at 72.4 GiB; KV cache = 128 KiB/token |
| 2 | Memory math | | `gpu_memory_utilization` | |
| 3 | Prefill vs decode | | input length | |