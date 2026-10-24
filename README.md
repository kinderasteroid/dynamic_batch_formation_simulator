# Dynamic Batch Formation Simulator

A Python-based **LLM inference scheduling simulator** that benchmarks three batching strategies: Fixed, Dynamic, and Continuous on a synthetic workload of 1,000 requests. Results are exported as CSVs and visual comparison graphs.

---

## Overview

Modern LLM serving systems must efficiently schedule incoming user requests to maximize GPU utilization and minimize latency. This simulator models the key scheduling decisions without needing a real GPU, using a lightweight time-stepped engine that mimics prefill and decode phases.

The simulator compares three strategies side-by-side:

| Strategy | Description |
|---|---|
| **Fixed Batching** | Waits until a full batch (e.g., 16 requests) is available before processing. Simple but introduces queuing delays. |
| **Dynamic Batching** | Processes whatever is available (up to a max size) without waiting for a full batch. Better latency under low load. |
| **Continuous Batching** | Immediately fills free slots as requests complete, without blocking. Closest to production systems like vLLM. |

---

## Algorithms Used & Why

### 1. Fixed Batching

**Algorithm:**
- Accumulate requests in a queue until exactly `batch_size` (default: 16) are available.
- Once full, run **prefill** (process all prompt tokens in parallel) then loop through **decode steps** until every request in the batch has generated its target number of tokens.
- Only then pick the next batch.

**Time complexity per batch:** `O(B × T_max)` where `B` = batch size, `T_max` = longest output in that batch.

**Why it exists:**  
Simple to implement and predictable in memory usage. Used as a **baseline** to highlight its core flaw: every request in the batch is blocked until the *slowest* one finishes, and the system sits idle waiting for a full batch under low load.

---

### 2. Dynamic Batching

**Algorithm:**
- At each scheduling tick, take *however many* requests are currently waiting (up to `max_batch_size`).
- Run prefill + decode on that variable-sized batch until all finish, then repeat.

**Time complexity per batch:** Same as fixed `O(B × T_max)`, but `B` varies (1 → max_batch_size).

**Why it's better than Fixed:**  
Eliminates the artificial queuing delay caused by waiting for a full batch. Under sparse load, single requests are processed immediately. Under heavy load, it naturally grows to full capacity.  
**Remaining flaw:** Still a "stop-the-world" approach, the batch must fully complete before new requests can be admitted.

---

### 3. Continuous Batching *(Iteration-level scheduling)*

**Algorithm:**  
This is the key innovation, scheduling happens at the **decode-step level**, not the batch level:

```
while requests pending or active:
    1. Fill empty slots with newly arrived requests (prefill them immediately)
    2. Run ONE decode step across ALL active sequences simultaneously
    3. Evict any sequences that just finished freeing their slot instantly
    4. Repeat
```

Each "slot" in the batch is recycled the moment a sequence completes, rather than waiting for the entire batch to finish.

**Why this wins:**
- **No head-of-line blocking** : a short request doesn't wait for a long one.
- **Maximum GPU utilization** : the active batch stays full at all times.
- **Lower TTFT** : newly arrived requests are prefilled immediately into freed slots.
- This is the core algorithm behind production systems like **vLLM**, **TensorRT-LLM**, and **Orca**.

---

### Workload Generation: Poisson Arrivals

Requests are generated using an **exponential inter-arrival time** (`random.expovariate(λ)`), which models a **Poisson process** , the standard model for independent, memoryless request arrivals in queuing theory. Prompt lengths and output lengths are drawn from uniform distributions to simulate realistic variance.

**Why Poisson?**  
Real web traffic is well-approximated by Poisson processes. Using exponential inter-arrivals ensures no artificial batching benefits from perfectly spaced requests.

---

### Engine Simulation Model

The `LLMEngine` uses a **time-stepped simulation** with constants derived from real hardware characteristics:

| Phase | Formula | Rationale |
|---|---|---|
| Prefill | `Σ(input_tokens) × 0.5ms` | Prefill is compute-bound; cost scales linearly with total prompt tokens |
| Decode step | `20ms + (n_active × 2ms)` | Decode is memory-bandwidth-bound; cost grows with batch size due to KV-cache reads |

This models the fundamental asymmetry between prefill (parallel, compute-bound) and decode (sequential per-token, memory-bound), the root cause of why batching strategy matters so much for LLM serving.


## How to Run

### Option 1: Local (Python)

**Prerequisites:** Python 3.11+

```bash
# 1. Clone this repo
cd dynamic_batch_formation_simulator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the experiment
PYTHONPATH=. python scripts/run_experiments.py
```

Results are saved to `./results/`.

---

### Option 2: Docker (Recommended)

**Prerequisites:** Docker + Docker Compose

```bash
# Build and run
docker compose up --build
```

> The container mounts `./results` so output files appear on your host machine automatically.

#### GPU Support (NVIDIA)
If you have an NVIDIA GPU, uncomment the GPU section in `docker-compose.yaml` and update `Dockerfile` to use:
```dockerfile
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir -r requirements.txt
```

---

## Outputs & What to Expect

After running, the `results/` folder contains:

### CSV Files (per-request metrics)

| File | Description |
|---|---|
| `fixed_metrics.csv` | Per-request stats for Fixed Batching |
| `dynamic_metrics.csv` | Per-request stats for Dynamic Batching |
| `continuous_metrics.csv` | Per-request stats for Continuous Batching |

Each CSV has columns: `request_id`, `arrival_time`, `start_time`, `first_token_time`, `end_time`, `input_tokens`, `output_tokens`, `ttft`, `e2e_latency`.

---

### Charts

#### 1. `ttft_comparison.png` : Average Time To First Token
Bar chart comparing average TTFT across all three strategies.
> **Lower is better.** Continuous batching typically wins here as new requests are admitted as soon as slots open.

#### 2. `throughput_comparison.png`: System Throughput
Bar chart of total tokens generated per second for each strategy.
> **Higher is better.** Continuous batching achieves the highest GPU utilization.

#### 3. `latency_cdf.png`: End-to-End Latency CDF
Cumulative distribution of e2e latency across all requests.
> **Steeper curve shifted left is better.** This shows tail-latency behaviour, critical for real-world SLA analysis.

---

##  Simulation Parameters

Tuneable in `scripts/run_experiments.py`:

| Parameter | Default | Effect |
|---|---|---|
| `num_requests` | 1000 | Total synthetic requests |
| `arrival_rate` | 5.0 req/s | Poisson arrival rate |
| `max_batch_size` | 16 | Max concurrent sequences |

Tuneable in `src/engine_wrapper.py`:

| Constant | Default | Represents |
|---|---|---|
| `time_per_prefill_token` | 0.5 ms | Cost per prompt token |
| `time_per_decode_step` | 20 ms | Base decode latency |
| `batch_penalty` | 2 ms/seq | Memory-bandwidth cost of larger batches |

---

##  Dependencies

| Package | Purpose |
|---|---|
| `llama-cpp-python` | LLM engine binding (GGUF model support) |
| `pandas` | Data processing and CSV export |
| `numpy` | Numerical utilities |
| `matplotlib` | Plot generation |
| `seaborn` | Styled statistical charts |

---

##  Key Takeaways

- **Fixed Batching** introduces unnecessary wait time when the queue is sparse.
- **Dynamic Batching** is more responsive but can still leave GPU slots idle mid-decode.
- **Continuous Batching** (used by vLLM, TensorRT-LLM) achieves the best throughput and TTFT by immediately recycling finished slots, especially under high concurrency.
