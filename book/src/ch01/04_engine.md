# The Engine

Every chapter of this book centres on one class: the engine. Its job is to accept a prompt string and return a `GenerateResult`. What changes from chapter to chapter is how it produces that result — which data structures it uses internally, how it schedules work, whether it caches past computation. What never changes is the interface the engine exposes.

This section introduces `BaselineEngine`, the Chapter 1 implementation, and the abstract base class it fulfils.

Run it:

```
python 04_engine.py
```

## The interface — `BaseOfflineEngine`

```python
class BaseOfflineEngine(ABC):

    @abstractmethod
    def generate(
        self,
        prompt:         str,
        max_new_tokens: int,
        params:         SamplingParams | None = None,
    ) -> GenerateResult: ...

    @abstractmethod
    def generate_batch(
        self,
        prompts:        list[str],
        max_new_tokens: int,
        params:         SamplingParams | None = None,
    ) -> list[GenerateResult]: ...
```

`BaseOfflineEngine` is an abstract base class — a class that can never be instantiated on its own but enforces that any subclass provides specific methods. In Python, this guarantee is expressed with the `@abstractmethod` decorator: attempting to instantiate a class that inherits from `BaseOfflineEngine` without implementing both `generate` and `generate_batch` raises a `TypeError` at the moment of construction, before any inference code runs. The error is caught early, close to where the contract was broken, rather than failing silently during a benchmark run.

The design choice of an abstract base class over a plain protocol or duck-typed interface matters for this book in particular. Each chapter defines its own engine class. The FastAPI server in `05_server.py` is built by a shared factory function (`make_app`) that accepts any `BaseOfflineEngine` and returns a working application. The benchmark scripts in `06_benchmark_offline.py` call `engine.generate()` directly. None of that code needs to know which chapter's engine it is talking to — the abstract base class is the contract that makes that substitution safe. This is the same pattern production inference systems use: vLLM's `LLMEngine`, SGLang's `LLM` class, and TGI's `TextGenerationInterface` are all variations of the same idea.

## Loading the model

```python
def __init__(self, model_name: str = MODEL_NAME):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model     = (
        AutoModelForCausalLM
        .from_pretrained(model_name, torch_dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )
```

`__init__` loads the model exactly once, when the engine is constructed. This matters because loading a multi-gigabyte checkpoint from disk and moving its weights to the GPU can take several seconds; doing it once and keeping the model resident in memory for the lifetime of the engine is the pattern every production system follows.

`torch_dtype=torch.float32` asks HuggingFace to load weights in 32-bit floating point. For a small teaching model this is fine. In production, you would typically pass `torch.float16` or `torch.bfloat16` to halve the memory footprint. Chapter 12 covers quantisation, which goes further still.

`.to(DEVICE).eval()` moves all weight tensors from CPU RAM to the target device and switches the model from training mode to inference mode. In training mode, certain layers — `BatchNorm`, `Dropout` — behave differently: dropout randomly zeros activations during the forward pass, and batch norm tracks running statistics. Calling `.eval()` disables both. For a transformer used purely for inference, `.eval()` is always correct. Omitting it is a common source of non-deterministic outputs that are difficult to debug.

## The generate method

```python
def generate(
    self,
    prompt:         str,
    max_new_tokens: int,
    params:         SamplingParams | None = None,
) -> GenerateResult:
    params    = params or SamplingParams()
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        # ← Chapter 2 replaces this with our own token-by-token loop
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=not params.is_greedy,
            temperature=params.temperature if not params.is_greedy else 1.0,
            top_p=params.top_p,
        )
    elapsed_s = time.perf_counter() - t0

    new_ids = output_ids[0][input_ids.shape[1]:]
    text    = self.tokenizer.decode(new_ids, skip_special_tokens=True)
    n_tok   = len(new_ids)
```

The generate method is five steps in one place: parameter defaulting, tokenisation, inference, output slicing, and metrics construction.

If the caller passes `None` for `params`, the line `params = params or SamplingParams()` substitutes the default `SamplingParams` object, which uses greedy decoding. This means callers that do not care about sampling get deterministic output without any extra code.

The prompt string is then encoded to a tensor of token IDs and sent to the same device as the model weights. `return_tensors="pt"` produces a PyTorch tensor with a batch dimension of one, so `input_ids` has shape `[1, prompt_length]`.

The `torch.no_grad()` context manager tells PyTorch to skip building the computation graph — the record of operations needed to compute gradients during backpropagation. During inference no gradient is ever needed, so building the graph would waste both memory and time. On a long prompt the difference is measurable. Forgetting `torch.no_grad()` is one of the most common sources of out-of-memory errors in inference code.

Inside the context, `self.model.generate()` runs HuggingFace's autoregressive loop. The loop feeds the model the growing sequence of tokens, one step at a time, extracts the logit vector for the last position, applies the requested sampling strategy — greedy argmax, temperature scaling, nucleus filtering — and appends the chosen token back onto the sequence. It repeats until `max_new_tokens` tokens have been generated or an end-of-sequence token appears. The entire loop is hidden inside a single function call. This is what makes Chapter 1 a black box: the loop exists, but we cannot see inside it. Chapter 2 rewrites it from scratch so that each step is visible.

The `do_sample` flag translates `SamplingParams.is_greedy` into HuggingFace's vocabulary: `False` means greedy, `True` enables multinomial sampling. When `do_sample=True`, the `temperature` and `top_p` parameters are forwarded from `SamplingParams`. When `do_sample=False`, temperature is forced to `1.0` because it has no effect in greedy mode and HuggingFace raises a warning if it is set to anything else.

After the call, `output_ids[0]` strips the batch dimension, giving a one-dimensional tensor of the complete sequence — prompt tokens followed by generated tokens. Slicing from `input_ids.shape[1]` forward isolates only the generated part. The prompt tokens were already shown to the user; they should not appear in the output string.

## Returning a GenerateResult

```python
    return GenerateResult(
        text       = text,
        output_ids = new_ids.tolist(),
        metrics    = RequestMetrics(
            throughput    = n_tok / max(elapsed_s, 1e-9),
            input_tokens  = input_ids.shape[1],
            output_tokens = n_tok,
        ),
    )
```

The `GenerateResult` envelope is constructed with the three fields introduced in `03_contracts.py`. `output_ids` calls `.tolist()` to convert the PyTorch tensor into a plain Python list, because `GenerateResult` is a plain dataclass that knows nothing about PyTorch and should not.

`RequestMetrics` is constructed with only the three required fields. `throughput` is `n_tok / elapsed_s` — output tokens divided by wall-clock seconds. The `max(elapsed_s, 1e-9)` guard prevents a division by zero on extremely fast machines or when the token count is very small. `ttft_ms` and `tpot_ms` are intentionally absent: HuggingFace runs prefill and decode as a single fused call, so the moment the first token was produced is invisible from outside `model.generate()`. Chapter 2 builds the loop step by step, which makes both timings accessible.

## generate_batch

```python
def generate_batch(
    self,
    prompts:        list[str],
    max_new_tokens: int,
    params:         SamplingParams | None = None,
) -> list[GenerateResult]:
    return [self.generate(p, max_new_tokens, params) for p in prompts]
```

`generate_batch` is a sequential loop over `generate`. Each prompt is processed one at a time, with no parallelism. This implementation satisfies the interface contract — the method exists and returns the right type — but it does not exploit any of the efficiency opportunities that batching makes possible. On a GPU, the compute units sit mostly idle while the model processes a single short sequence; running multiple sequences in parallel would keep them busy and amortise the fixed cost of a forward pass across many tokens simultaneously. Chapter 5 replaces this loop with true static batching. Chapter 9 extends that to continuous batching, which handles requests of different lengths arriving at different times.

## Running the engine

When run as a script, `04_engine.py` constructs the engine, runs one greedy generation call, and prints the result:

```
[engine] loading arnir0/Tiny-LLM on cpu ...
[engine] ready

[Prompt]
  According to all known laws of aviation, ...

[Output]
  The bee flies because it does not know it cannot.

[Metrics]
  throughput      12.3
  input_tokens    18
  output_tokens   40
```

The engine is now a fully functional unit. The next step is to wrap it in an HTTP server so that any client — a benchmark script, a web UI, another service — can call it over the network using a standard API.
