<!-- >/dev/null 2>&1
exit 0
-->

# vLLM Performance Benchmarking Sweep / run_sweep.sh script

This bash script automates a performance benchmarking sweep for a running vLLM inference server. It tests how the server handles increasing request loads (using the `Llama-3.1-8B` model) while simultaneously recording GPU hardware metrics.

## Technical Breakdown

### 1. Setup and Configuration

* `set -e`: Ensures the script aborts immediately if any command fails.
* **Variables**:
  * `OUT`: Sets the output path for results, specifically labeled for an A100 80GB GPU.
  * `IMG`: Specifies the vLLM Docker image to use for the benchmark client.
  * `MODEL`: Defines the target model (`meta-llama/Llama-3.1-8B-Instruct`).

### 2. The `run` Function

This function executes a single benchmark trial. It takes three arguments: `$1` (output folder label), `$2` (request rate), and `$3` (random seed).

* **GPU Monitoring**: `nvidia-smi dmon` runs in the background (`&`) to poll GPU #2 for Power (`p`), Utilization (`u`), Clocks (`c`), and Memory (`m`) metrics, logging them to `dmon.txt` with timestamps (`-o T`).
* **Benchmark Client**: It spins up a Docker container using the host network (to connect to the already-running server on port 8000).
* **Workload Parameters**: It runs `vllm bench serve` with synthetic traffic:
  * 200 total prompts per run (`--num-prompts 200`).
  * Fixed sizes of 512 input tokens and 128 output tokens (`--random-input-len 512 --random-output-len 128`).
  * `--ignore-eos`: Forces the model to generate exactly 128 tokens by ignoring the "end of sequence" token, ensuring a consistent compute workload.
  * Results and logs are saved to the mounted output directory.
* **Cleanup**: `kill $dmon` stops the background GPU monitor once the Docker command finishes.

### 3. Execution Flow

* **Warmup**: `run warmup/run1 1 42` sends 1 request per second to initialize the server (loading CUDA graphs, warming up memory pools) so subsequent tests aren't skewed.
* **Baseline (1 req/sec)**: Runs two distinct trials (seeds 43 and 44) at a low rate.
* **Stress Test Loop**: The `for` loop scales the load up to 4, 16, 32, and 64 requests per second, finishing with `inf` (infinite rate, sending all 200 requests concurrently to test maximum throughput). It runs two separate trials for each rate to ensure statistical consistency.

