# Inference From Scratch

Build a complete LLM inference engine from scratch — in Python, in plain PyTorch, one concept at a time.

📖 **Read the book:** https://arunpatala.github.io/inference-from-scratch-book/

---

## What you will build

A single serving engine that evolves chapter by chapter. Each chapter adds one optimization, measures the speedup, and swaps it into the running server in one line.

By the end of Chapter 13, your engine handles autoregressive generation, KV caching, continuous batching, flash attention, paged attention, prefix caching, quantization, and speculative decoding — a simplified vLLM, built entirely by you.

## Status

🚧 Work in progress. Chapters releasing incrementally.

| # | Chapter | Status |
|---|---------|--------|
| 1 | What does an inference engine look like? | 🚧 Coming soon |
| 2 | How does generation actually work? | 🚧 Coming soon |
| 3 | What is the model doing? | 🚧 Coming soon |
| 4 | Why is decode slow? (KV cache) | 🚧 Coming soon |
| 5 | How do we serve multiple requests? (Static batching) | 🚧 Coming soon |
| 6 | Profiling — where is time actually spent? | 🚧 Coming soon |
| 7 | Writing GPU kernels with Triton | 🚧 Coming soon |
| 8 | How do we make attention faster? (Flash Attention) | 🚧 Coming soon |
| 9 | Continuous batching | 🚧 Coming soon |
| 10 | Memory management (Paged Attention) | 🚧 Coming soon |
| 11 | Prefix caching | 🚧 Coming soon |
| 12 | Quantization (INT4/GPTQ) | 🚧 Coming soon |
| 13 | Speculative decoding | 🚧 Coming soon |
