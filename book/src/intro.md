# Inference From Scratch

> **This book is a work in progress. Chapters are being released incrementally.**

---

## What is this book?

Most engineers who work with language models treat them as black boxes: send a prompt, get a
response. This book opens every box.

You will build a complete LLM inference engine from scratch — in Python, in plain PyTorch, one
concept at a time. By the end, you will have written your own autoregressive loop, your own
attention kernel, your own KV cache, your own scheduler, your own batching system, and your own
speculative decoder. Every optimization will be measured before and after against the same HTTP
benchmark harness you build in Chapter 1.

The engine you end up with is a simplified vLLM, built entirely by you, line by line.

---

## Who is this for?

- ML engineers who use inference systems but want to understand what is happening inside them
- Systems engineers curious about what makes LLM serving different from regular serving
- Researchers who want to implement and experiment with inference optimizations
- Anyone who has read "Attention is All You Need" and wants to see how that attention becomes
  a production serving system

**Prerequisites:** Python, basic PyTorch, some familiarity with neural networks. No CUDA or GPU
programming experience required — that is covered from scratch.

---

## What you will build

A single serving engine that evolves chapter by chapter. Each chapter adds one optimization,
measures the speedup, and swaps it into the running server in one line. The benchmark numbers
are real — run on the same hardware, the same model, the same prompts.

By the end of Chapter 13, your engine handles:

- ✅ Autoregressive generation with your own loop
- ✅ Full transformer model built from scratch with real weights loaded
- ✅ KV cache eliminating redundant computation
- ✅ Static and continuous batching for multi-request throughput
- ✅ Flash attention reducing memory from O(n²) to O(n)
- ✅ Paged attention eliminating memory fragmentation
- ✅ Prefix caching reusing KV blocks across requests
- ✅ INT4 quantization halving memory footprint
- ✅ Speculative decoding generating multiple tokens per step

---

## Chapters

| # | Question | Key concept | Status |
|---|----------|-------------|--------|
| 1 | What does an inference engine look like? | Baseline HF server, TTFT/TPOT/throughput | 🚧 Coming soon |
| 2 | How does generation actually work? | Autoregressive loop, prefill vs decode | 🚧 Coming soon |
| 3 | What is the model doing? | Transformer from scratch, load real weights | 🚧 Coming soon |
| 4 | Why is decode slow, and how do we fix it? | KV cache | 🚧 Coming soon |
| 5 | How do we serve multiple requests at once? | Static batching, scheduler | 🚧 Coming soon |
| 6 | Profiling — where is time actually spent? | Roofline model, memory vs compute bound | 🚧 Coming soon |
| 7 | Writing GPU kernels with Triton | GEMM, tiling, shared memory | 🚧 Coming soon |
| 8 | How do we make attention faster? | Online softmax, Flash Attention | 🚧 Coming soon |
| 9 | Continuous batching | Iteration-level scheduling, no padding waste | 🚧 Coming soon |
| 10 | How do we manage memory efficiently? | Paged attention, block tables | 🚧 Coming soon |
| 11 | How do we avoid recomputing common prefixes? | Prefix caching, hash-based block reuse | 🚧 Coming soon |
| 12 | How do we run bigger models on less memory? | INT4/GPTQ quantization | 🚧 Coming soon |
| 13 | How do we generate faster without changing the model? | Speculative decoding, draft-target | 🚧 Coming soon |

---

## The code

Every chapter has a companion directory of numbered Python files. Each file is self-contained and
runnable. The files build on each other within the chapter — run them in order.

```
chapters/
├── ch01_baseline/
│   ├── 00_start.py        ← run this first
│   ├── 01_load.py
│   ├── 02_sampling.py
│   ├── 03_server.py
│   └── 04_benchmark.py
├── ch02_generate_loop/
│   └── ...
```

Clone the repo and run any chapter independently:

```bash
git clone https://github.com/arunpatala/inference-from-scratch.git
cd inference-from-scratch/CODE/chapters/ch01_baseline
pip install torch transformers fastapi uvicorn requests
python 00_start.py
```

---

## A note on the model

All examples use small, freely available models that run on CPU so you can follow along without a
GPU. The optimizations are GPU-native — the book explains what each one does on GPU hardware and
why it matters at scale — but the code runs and produces correct output on any machine.

---

*Built in public. Feedback and corrections welcome via GitHub issues.*
