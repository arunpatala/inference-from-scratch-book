# Contracts

Before wiring an engine into a server, or a server into a benchmark, it helps to agree on what data flows between them. In software, that agreement is called a contract — a definition of the types that cross a boundary, making it possible for each side to be written, tested, and replaced independently.

The three types introduced in this section appear in every single chapter of this book. They are defined once in `shared/` and never change. What changes from chapter to chapter is the engine that implements them. Chapter 2 replaces the HuggingFace call inside `generate()` with our own loop. Chapter 4 adds a KV cache. Chapter 9 adds continuous batching. In every case, the surrounding code — the server, the benchmark harness, the metrics display — stays identical because the contracts stay identical.

```python
from shared.metrics import RequestMetrics
from shared.result import GenerateResult
from shared.sampling_params import SamplingParams
```

Run it:

```
python 03_contracts.py
```

## SamplingParams — how should tokens be chosen?

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k:       int   = -1
    top_p:       float = 1.0
    ignore_eos:  bool  = False
    max_tokens:  int   = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

`SamplingParams` is a dataclass that carries the sampling configuration for a single request. It travels from the caller of `generate()` down through the engine to the sampler that picks each output token.

The defaults are deliberately conservative. `temperature=0.0` means greedy decoding — always take the most probable token, no randomness. `top_k=-1` means top-k filtering is disabled. `top_p=1.0` means all tokens remain in the candidate pool. Together, these defaults produce deterministic, reproducible output with a single line:

```python
params = SamplingParams()   # greedy
```

To switch to nucleus sampling with moderate creativity, the caller constructs:

```python
params = SamplingParams(temperature=0.8, top_p=0.95)
```

The `is_greedy` property gives every engine a single place to check whether randomness is needed before calling the sampler, avoiding the cost of multinomial sampling when the result is predetermined. If both `top_p` is 1.0 and temperature is zero (or top_k is 1), the distribution has already collapsed to a point mass and sampling adds nothing.

The reason to put these parameters in a dataclass rather than passing them as individual keyword arguments is consistency. As the number of sampling strategies grows — minimum-p sampling, repetition penalty, beam search — the function signature stays the same. The caller constructs a `SamplingParams` object; the engine receives it; the sampler reads from it. Adding a new field to the dataclass does not break any existing call site.

## RequestMetrics — how do we measure performance?

```python
@dataclass
class RequestMetrics:
    throughput:    float
    input_tokens:  int
    output_tokens: int
    ttft_ms:       float = 0.0   # 0.0 = not measured
    tpot_ms:       float = 0.0   # 0.0 = not measured
```

`RequestMetrics` records what happened during a single `generate()` call. It lives inside every `GenerateResult` and is serialised into the server's HTTP response under a `metrics` key.

The three required fields are always present. `throughput` is output tokens per second — the most direct measure of how fast the engine is running. `input_tokens` and `output_tokens` are the prompt and completion lengths, which together determine how much work was done and provide the numbers that go into the OpenAI-compatible `usage` field of the response.

The two optional fields — `ttft_ms` and `tpot_ms` — default to `0.0` and are only included in the serialised output when non-zero. TTFT (time to first token) measures how long the prefill phase took: the delay between receiving the request and producing the first output token, which is what the user experiences as initial latency. TPOT (time per output token) measures the average duration of each decode step, which determines how fast text streams to the user once generation has begun.

In Chapter 1, both fields stay at zero. HuggingFace runs prefill and decode as a single opaque call, so there is no way to separate them from the outside. Chapter 2 builds the loop step by step, at which point timing each phase becomes straightforward and both fields get populated. The `0.0` default is not a placeholder — it is a signal that the measurement was not available, and `as_dict()` is designed to omit it cleanly:

```python
def as_dict(self) -> dict:
    d = {
        "throughput":    round(self.throughput, 1),
        "input_tokens":  self.input_tokens,
        "output_tokens": self.output_tokens,
    }
    if self.ttft_ms:
        d["ttft_ms"] = round(self.ttft_ms, 2)
    if self.tpot_ms:
        d["tpot_ms"] = round(self.tpot_ms, 2)
    return d
```

A Ch01 engine produces `{"throughput": 12.5, "input_tokens": 20, "output_tokens": 40}`. A Ch02 engine produces the same dict extended with `"ttft_ms"` and `"tpot_ms"`. The server code that formats the response never needs to know which chapter it is running; it calls `result.metrics.as_dict()` and gets back whatever the engine was able to measure.

## GenerateResult — the envelope

```python
@dataclass
class GenerateResult:
    text:       str
    output_ids: list[int]
    metrics:    RequestMetrics
```

`GenerateResult` is the return type of every `generate()` call in the book. It is deliberately minimal: three fields, no methods, no logic.

`text` is the decoded output string — what gets shown to the user or returned in the HTTP response body. `output_ids` is the raw list of generated token IDs, not including the prompt tokens. Keeping the raw IDs alongside the text matters because some later chapters — speculative decoding in Chapter 13, prefix caching in Chapter 11 — need to reason about the token sequence directly, not just the string it decodes to. `metrics` holds a `RequestMetrics` instance, or any subclass of it.

That last point — "or any subclass" — is what makes the design extensible. Chapter 13 introduces `SpecDecodeMetrics`, which inherits from `RequestMetrics` and adds two additional fields:

```python
@dataclass
class SpecDecodeMetrics(RequestMetrics):
    tokens_proposed: int = 0
    tokens_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.tokens_accepted / max(self.tokens_proposed, 1)
```

A Chapter 13 engine returns a `GenerateResult` with a `SpecDecodeMetrics` inside it. The server still calls `result.metrics.as_dict()`, and because `SpecDecodeMetrics` overrides `as_dict()` to include the extra fields, the acceptance rate appears in the response automatically. No code outside the engine changes.

## The data flow

Putting the three types together:

```
prompt (str)
    │
    ▼
engine.generate(prompt, max_new_tokens, SamplingParams)
    │
    ▼
GenerateResult
    ├── text        → returned to caller / shown in HTTP response
    ├── output_ids  → token IDs, kept for chapters that need them
    └── metrics     → RequestMetrics.as_dict() → "metrics" in JSON
```

Every chapter from here to the end of the book follows this shape. The prompt goes in, a `GenerateResult` comes out. What varies is everything that happens in between — and the contracts above are what make it possible to swap that middle out without touching anything else.
