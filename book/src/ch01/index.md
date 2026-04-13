# What does an inference engine look like?

An inference engine is a surprisingly small thing. Strip away the HTTP routing, the batching logic, the memory management, and the GPU kernels, and what remains is a program that does three things: it takes text in, runs it through a neural network, and returns text out. That loop, at its core, is what this entire book is about.

This chapter builds the simplest possible version of that loop in one sitting. By the end you will have a working server that accepts prompts over HTTP and returns generated text, along with a benchmark harness to measure how fast it runs. Every subsequent chapter takes one component from this baseline and replaces it with something faster, more capable, or more principled — but the overall shape stays the same.

The chapter is organised as a sequence of self-contained files, each one adding a single layer of understanding. Read them in order, run each one, and by the time you reach the benchmark you will have a clear mental model of what an inference engine actually does — and exactly where the interesting engineering problems live.

| File | What it covers |
|---|---|
| `00_start.py` | The complete engine in 35 lines |
| `01_load.py` | What model loading is actually doing |
| `02_sampling.py` | How temperature, top-k, and top-p change the output |
| `03_contracts.py` | The shared types every chapter uses |
| `04_engine.py` | The engine abstraction |
| `05_server.py` | Wrapping the engine in an HTTP API |
| `06_benchmark_offline.py` | Measuring throughput directly |
| `06_benchmark_online.py` | Measuring throughput over HTTP |
