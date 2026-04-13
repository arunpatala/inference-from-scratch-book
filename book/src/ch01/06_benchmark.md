# The Benchmark

Knowing that an engine produces correct output is not enough. Every chapter of this book introduces a new engine implementation, and the only way to know whether a change is actually faster is to measure it systematically, with the same methodology, against the same numbers. This section builds the measurement tooling and produces the baseline numbers that every subsequent chapter will try to beat.

There are two benchmark scripts. `06_benchmark_offline.py` calls the engine directly in-process. `06_benchmark_online.py` sends HTTP requests to a running server. Both produce the same table of numbers; the difference between them is the overhead of the network stack.

Run offline:

```
python 06_benchmark_offline.py
```

Run online (server is launched and stopped automatically):

```
python 06_benchmark_online.py
```

## What to measure

A benchmark number is only useful if you understand what it is measuring and what it is not. Three quantities matter for a single-user, single-request system like Chapter 1:

**Throughput (tokens per second)** is the rate at which the model produces output tokens. For a generative model, this is `output_tokens / elapsed_seconds`. It is the primary number that every chapter optimisation targets. Higher is better.

**Requests per second** is `1 / avg_wall_seconds`. For a single-threaded server, this is simply the reciprocal of average latency. It becomes independently interesting in Chapter 9, where continuous batching allows multiple requests to overlap, decoupling throughput from single-request latency.

**Wall time per request** is the total time from calling `engine.generate()` (or sending the HTTP request) to receiving the complete result. It is the number the end user experiences as latency. Because Chapter 1 generates all tokens before returning, wall time grows linearly with `max_new_tokens`.

**Time-to-first-token (TTFT)** and **time-per-output-token (TPOT)** are absent from Chapter 1. HuggingFace's `model.generate()` runs prefill and decode as a single fused call and returns only after all tokens are produced; the moment the first token was generated is not observable from outside the call. Chapter 2 builds the loop step by step and populates both. The benchmark infrastructure tracks them in `BenchStats` with `0.0` defaults and omits them from the printed table until they are non-zero.

## Warmup runs

```python
for _ in range(n_warmup):
    engine.generate(prompt, max_tokens, params)
```

The first few calls to `engine.generate()` are always slower than subsequent ones. Several independent effects cause this. PyTorch compiles CUDA kernels lazily: the first time a particular operation is encountered with a particular tensor shape, PyTorch compiles and caches a CUDA kernel for it. That compilation can take tens or hundreds of milliseconds. On CPU, the OS page-fault mechanism brings model weight pages into RAM only as they are accessed; the first forward pass touches every weight page for the first time. Python's import machinery and interpreter warm up. On subsequent calls, compiled kernels are cached, weights are resident in memory, and the interpreter is in a steady state.

Warmup runs are discarded. They are not included in any computed average. The number of warmup runs is configured by `WARMUP_RUNS` in `shared/config.py` (default: 3), which is enough to get past both kernel compilation and page faults on the models used in this book. Production benchmarks like those published by NVIDIA NIM use a dedicated warmup phase for the same reason, though at larger scale they may require dozens of iterations.

## BenchStats

```python
@dataclass
class BenchStats:
    req_per_s:     float
    tok_per_s:     float
    wall_ms:       float
    output_tokens: float
    n_runs:        int
    ttft_ms:       float       = 0.0
    tpot_ms:       float       = 0.0
    wall_times:    list[float] = field(default_factory=list)
```

`BenchStats` aggregates all measurements from a benchmark run into one object. `req_per_s`, `tok_per_s`, and `wall_ms` are averages over `n_runs` measured requests. `wall_times` keeps the raw per-request timings; later chapters that analyse latency distributions — p50, p95, p99 — can compute percentiles from this list without re-running the benchmark. `ttft_ms` and `tpot_ms` default to `0.0` and remain at zero when not populated, following the same optional-field convention used in `RequestMetrics`.

## Offline benchmark

```python
for _ in range(n_runs):
    t0      = time.perf_counter()
    result  = engine.generate(prompt, max_tokens, params)
    wall_ms = (time.perf_counter() - t0) * 1000

    wall_times.append(wall_ms)
    tok_counts.append(result.metrics.output_tokens)
    if result.metrics.ttft_ms:
        ttfts.append(result.metrics.ttft_ms)
    if result.metrics.tpot_ms:
        tpots.append(result.metrics.tpot_ms)
```

`offline_benchmark` calls `engine.generate()` directly. `time.perf_counter()` measures wall-clock elapsed time with nanosecond resolution — it is the correct timer for short durations. `time.time()` uses the system clock and can jump backward or forward if the OS adjusts it; `time.perf_counter()` is monotonic and immune to that. The multiplication by 1000 converts seconds to milliseconds, which are the natural unit for per-request latency on the models used in this book.

The output token count comes from `result.metrics.output_tokens` rather than re-counting `result.output_ids`. The engine already measured this; there is no reason to recompute it. `ttft_ms` and `tpot_ms` are collected only when non-zero, preserving the same `0.0`-signals-unmeasured convention used throughout the contracts.

Token counts matter because `max_new_tokens` is a ceiling, not a guarantee. HuggingFace's `model.generate()` stops early if it produces an end-of-sequence token before reaching `max_new_tokens`. If the average output is 35 tokens when you asked for 40, computing tokens-per-second using 40 would give a number that is 14% optimistic. Reading `output_tokens` from the result gives the correct denominator.

Aggregation divides total output tokens by total wall time:

```python
avg_wall   = _avg(wall_times)       # ms
avg_tokens = _avg(tok_counts)
req_per_s  = 1000 / avg_wall        # 1 / avg_seconds
tok_per_s  = avg_tokens / (avg_wall / 1000)
```

## Online benchmark

```python
t0   = time.perf_counter()
r    = requests.post(endpoint, json=payload)
r.raise_for_status()
wall_ms = (time.perf_counter() - t0) * 1000

data = r.json()
wall_times.append(wall_ms)
tok_counts.append(data["usage"]["completion_tokens"])
```

`online_benchmark` sends HTTP POST requests to the running server. The wall time is measured around the full `requests.post()` call — from the moment the client sends the request to the moment the response body is fully received. This includes TCP connection time (negligible for localhost), HTTP framing, JSON serialisation on the server, and JSON deserialisation on the client. The difference between this number and the offline wall time is the overhead the network stack and server layer add. For a localhost server this overhead is typically 0.5–5ms per request.

The token count is read from `data["usage"]["completion_tokens"]` — the OpenAI-standard field in the response. The non-standard `metrics` field is also read for `ttft_ms` and `tpot_ms` when present, using `.get()` with a fallback to an empty dict so that the benchmark works correctly against servers that do not include a `metrics` field.

`r.raise_for_status()` converts any HTTP 4xx or 5xx response into a Python exception immediately. Without it, a server error would silently produce `None` JSON and the benchmark would either crash with an unhelpful `KeyError` or, worse, record a zero-token measurement and silently corrupt the averages.

## Server lifecycle

```python
proc = launch_server(
    server_module = "05_server:app",
    port          = SERVER_PORT,
    health_url    = f"{SERVER_URL}/health",
    cwd           = CHAPTER_DIR,
)
try:
    stats = online_benchmark(SERVER_URL, ...)
    print_results("Ch01 baseline — online (HTTP)", stats)
finally:
    stop_server(proc)
```

`launch_server` starts uvicorn as a child process using `subprocess.Popen`. The parent process then polls `GET /health` in a loop, sleeping one second between attempts, until the server returns HTTP 200 or a 60-second timeout expires. The poll loop exists because uvicorn takes a variable amount of time to start: it must import the server module, which triggers model loading, which on CPU can take several seconds. Polling `/health` is more reliable than sleeping a fixed duration.

`stop_server` calls `proc.terminate()`, which sends `SIGTERM` to the uvicorn process, followed by `proc.wait()`, which blocks until the process exits. This ensures the port is fully released before the benchmark script itself exits. Without `proc.wait()`, the port can remain bound briefly after the script ends, causing the next run to fail with "address already in use".

The `try / finally` block around the benchmark is critical. If `online_benchmark` raises an exception — a server error, a network timeout, a keyboard interrupt — `stop_server` still runs in the `finally` branch. Without it, a failed benchmark would leave a uvicorn process running in the background, holding the port and the GPU memory.

## The printed table

```
  ──────────────────────────────────────────────────────
  Ch01 baseline — offline (no HTTP)
  ──────────────────────────────────────────────────────
  Req/s        : 0.82
  Tok/s        : 32.6
  Wall time    : 1224ms / request
  Tokens out   : 39 (avg over 20 runs)
  ──────────────────────────────────────────────────────
```

`print_results` omits `TTFT` and `TPOT` lines when they are zero, keeping the Chapter 1 output clean. Future chapters print a longer table. The label argument identifies which chapter and mode produced the table, making it straightforward to compare results across chapters or across offline and online runs side by side.

These numbers are the baseline. Chapter 2 prints the same table immediately after building its own token-by-token loop. If the Chapter 2 numbers match Chapter 1's to within a few percent, the new loop is correct — it is doing the same computation. If they differ significantly, there is a bug. The benchmark is the test.
