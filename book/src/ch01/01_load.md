# Loading a model

The first two lines of `00_start.py` are the most expensive lines in the entire program.

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
```

On a cold start, these two calls download several files from the HuggingFace Hub, parse a configuration file, construct a neural network architecture in memory, and then copy hundreds of millions of numbers into that architecture one tensor at a time. On subsequent runs the files are cached locally, so only the last two steps repeat — but even then, loading a large model can take several seconds. Understanding exactly what is happening in this time is the subject of this section.

Run:

```
python 01_load.py
python 01_load.py --model HuggingFaceTB/SmolLM2-135M
```

## What `AutoTokenizer.from_pretrained` downloads

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

A tokenizer is not a neural network — it is a deterministic lookup table combined with a set of rules. `from_pretrained` downloads two things. The first is a vocabulary file: a mapping from every token the model knows (typically 32,000 to 128,000 subword strings) to an integer index. The second is a configuration file that specifies which tokenization algorithm to use — Byte Pair Encoding (BPE), WordPiece, Unigram, or one of several others — along with any special tokens the model was trained with, such as the beginning-of-sequence marker or the padding token.

The `Auto` prefix means that HuggingFace inspects the configuration file and automatically selects the correct tokenizer class for the model you asked for. If the model was trained with a BPE tokenizer, you get a `BPETokenizer`. If it uses a SentencePiece model, you get that instead. You do not need to know which type is being used; the `encode` and `decode` methods work identically regardless.

Once loaded, the tokenizer is fast. It runs entirely on CPU, requires no GPU, and processes most inputs in under a millisecond. Its output — a tensor of integer indices — is small: a 200-word prompt might produce 250 token IDs, each one a 32-bit integer.

## What `AutoModelForCausalLM.from_pretrained` does

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
```

This call does three things in sequence: it reads the model's configuration, constructs the neural network architecture in memory, and loads the trained weights into it.

The configuration file — typically `config.json` in the model repository — describes the shape of the network: how many layers it has, how wide those layers are, how many attention heads each layer contains, what its vocabulary size is, and which activation function to use. `AutoModelForCausalLM` reads this file, looks up the right model class in an internal registry (mapping "llama" in the config to `LlamaForCausalLM`, for instance), and calls its constructor. At this point the network exists in memory but its parameters are random or uninitialized — it has the right shape, but no knowledge.

The second step copies the trained parameters into that skeleton. The weights are stored on disk as a `safetensors` file — a format designed to be faster and safer than the legacy `.pkl` format. Each tensor in the file is matched by name to a parameter in the model and copied in. For a 10M parameter model this takes milliseconds. For a 70B parameter model it takes minutes — each parameter is a 32-bit float requiring 4 bytes of memory, so 70 billion parameters occupy 280 gigabytes in `float32`. In practice, production models are almost always loaded in `float16` or `bfloat16` (2 bytes per parameter), halving that to 140 GB. Quantisation to 8-bit or 4-bit reduces it further still, and Chapters 12 and 13 address this directly.

The `torch_dtype=torch.float32` argument we pass in the chapter code forces all weights to be stored as 32-bit floats. This makes the arithmetic easier to reason about — the numbers are exactly representable, without the rounding that lower-precision formats introduce — at the cost of using twice the memory of `float16`. For the small model used in this chapter, the difference is negligible.

## Placing the model on a device

```python
model = model.to(DEVICE).eval()
```

PyTorch models and tensors always live on a specific device — either the CPU or a particular GPU. When a model is first loaded with `from_pretrained`, it lands on CPU. The call `.to(DEVICE)` moves every parameter tensor to the target device. If `DEVICE` is `"cuda"`, each tensor is copied from system RAM into GPU VRAM over the PCIe bus. If `DEVICE` is `"cpu"`, nothing moves — the model stays where it is.

The call `.eval()` switches the model into inference mode. During training, certain layers behave differently than they do at inference time. Dropout layers randomly zero out activations to prevent overfitting; at inference time, they should pass their input through unchanged. Batch normalisation layers use per-batch statistics during training and running statistics during inference. Calling `.eval()` activates the inference behaviour in all such layers. Omitting this call produces subtly wrong results that can be difficult to diagnose, because the model will still run — it just will not run correctly.

The two calls are chained because `.to()` returns the model itself, making it idiomatic to write them on a single line.

## Measuring what you loaded

```python
params = sum(p.numel() for p in model.parameters())
print(f"  parameters : {params / 1e6:.1f}M")
```

Every `nn.Parameter` in the model is a tensor. `p.numel()` returns the total number of elements in that tensor — for a weight matrix of shape `[4096, 4096]` that is 16,777,216 elements. Summing this across all parameters gives the total parameter count. Dividing by one million and formatting to one decimal place gives the familiar shorthand: "10.5M", "135M", "7B".

Parameter count is a useful proxy for model capability and cost, but it is not the whole story. Memory consumption also depends on the data type: a 7B parameter model in `float16` occupies 14 GB, but the same model in `float32` occupies 28 GB. GPU memory also includes space for the KV cache during generation — the amount of memory needed to store the intermediate attention keys and values for every token in every layer — which can rival the model weights in size for long contexts. Chapter 4 covers the KV cache in detail.

## Verifying it works: a tokenizer smoke test

```python
sample = "Hello, world!"
ids    = tokenizer.encode(sample, return_tensors="pt")
print(f"  text       : '{sample}'")
print(f"  token ids  : {ids.tolist()}")
print(f"  decoded    : '{tokenizer.decode(ids[0])}'")
```

Before running any generation, it is worth confirming that the tokenizer is working correctly. This snippet encodes a short string to token IDs and then decodes it back. If the decoded string matches the input — modulo any normalisation the tokenizer applies — everything is wired up correctly. It is a cheap sanity check that catches common mistakes like loading the wrong tokenizer for a given model, which produces plausible-looking but incorrect token IDs.

The round-trip `encode → decode` should be lossless for any string the tokenizer supports. Punctuation and whitespace can sometimes look different after the round-trip because the tokenizer uses a byte-level representation internally, but the semantic content should be identical.

## What comes next

At this point, the model and tokenizer are loaded, placed on the correct device, and verified to be working. The next section, `02_sampling.py`, keeps the model exactly as it is and instead explores how changing the sampling parameters — `temperature`, `top_k`, and `top_p` — changes the character of the generated text. The model is not the variable; the sampling strategy is.
