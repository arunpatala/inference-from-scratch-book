# Start here

This is the entire inference engine.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, PROMPT

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_text(prompt, model, tokenizer, max_length=512, temperature=1, top_k=50, top_p=0.95):
    inputs  = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_text(PROMPT, model, tokenizer))
```

Run it:

```
python 00_start.py
```

You will see a pause, then some text. That pause is the entire cost of being an AI: loading hundreds of millions of numbers from disk into memory, arranging them into matrices, and preparing to multiply them together at high speed. The text that follows is the result of doing exactly that, once per token, until the model decides it is done.

Every line of the program above is currently a black box. That is intentional. The goal of this first pass is to see the whole thing working before we start taking it apart. The rest of this chapter, and the rest of this book, opens each box in turn.

## What just happened

There are three steps inside `generate_text`, and they are the same three steps inside every language model inference system ever built, from the smallest research prototype to the largest production cluster.

### Step 1: Tokenization

```python
inputs = tokenizer.encode(prompt, return_tensors="pt")
```

A language model cannot read text. What it reads is integers — specifically, indices into a fixed vocabulary of tens of thousands of subword units called tokens. The tokenizer's job is to convert the raw string into that sequence of integers.

The word "tokenizer" hides a surprising amount of machinery. Before a single integer is produced, the input text is normalised (Unicode is standardised, whitespace is cleaned), then split into preliminary chunks on word boundaries and punctuation, then passed through a learned compression algorithm — typically Byte Pair Encoding or a variant of it — that merges common character sequences into single tokens. A word like "unbelievable" might become three tokens: "un", "believ", "able". A rare technical term might be split into individual characters. The choice of how to split determines how efficiently the model uses its fixed context window, and different models make different choices.

The result of `tokenizer.encode` is a PyTorch tensor of integers. The sentence "According to all known laws of aviation, there is no way a bee should be able to fly" becomes something like `[12902, 2311, 477, 3245, 4141, ...]` — the exact values depend on the model's vocabulary. This is what the model sees. It has never seen the characters 'b', 'e', 'e' in sequence; it has seen the integer that represents the token "bee". The argument `return_tensors="pt"` wraps those integers in a PyTorch tensor so they can be passed directly to the model.

### Step 2: Generation

```python
outputs = model.generate(
    inputs,
    max_length=max_length,
    temperature=temperature,   # controls randomness (0 = deterministic)
    top_k=top_k,               # only consider top-k most likely tokens
    top_p=top_p,               # only consider tokens in top-p probability mass
    do_sample=True,
)
```

This single call is where the interesting work happens, and also where the most is hidden. Inside `model.generate`, the model runs a two-phase process that will occupy several later chapters.

The first phase is called **prefill**. The model takes the full sequence of input token IDs and runs them through all of its layers in a single parallel forward pass. Every token attends to every earlier token simultaneously, building up rich contextual representations. The output of prefill is a probability distribution over the vocabulary: given this prompt, how likely is each possible next token? This phase is compute-intensive because it processes all tokens at once, and its cost grows with the length of the input.

The second phase is called **decode**. The model samples one token from that distribution, appends it to the sequence, and then runs another forward pass — but this time processing only the new token, because the representations of all earlier tokens were computed in prefill and can be reused. The model outputs another distribution, another token is sampled, and the loop continues. Each decode step is cheap compared to prefill, but the steps must happen sequentially: you cannot start computing token five until token four has been chosen. This sequential constraint is the fundamental reason why inference is hard to scale, and it is the thread that runs through most of the optimisations in later chapters.

The parameters `temperature`, `top_k`, and `top_p` all control the sampling step — how the model chooses which token to pick from the distribution at each decode step. At `temperature=0` the choice is deterministic: always take the most likely token, which is called greedy decoding. At higher temperatures the distribution is flattened, making lower-probability tokens more likely and the output more varied. `do_sample=True` tells HuggingFace to use sampling rather than greedy search. Section 1.2 explores all three parameters in detail.

### Step 3: Detokenization

```python
return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

The third step is the inverse of the first. `model.generate` returns a tensor of token IDs that includes both the original prompt and the newly generated tokens. `outputs[0]` selects the first (and only) sequence in the batch. `tokenizer.decode` maps each integer back to its string representation and joins them into a single string. The argument `skip_special_tokens=True` strips out control tokens like the model's end-of-sequence marker, which should not appear in the final output.

The full pipeline, then, is: text → integers → neural network loop → integers → text. Everything else in this book is engineering that makes the middle step faster, cheaper, or more capable of handling many requests at once.

## What to notice when you run it

There are two pauses. The first, before any output appears, is the model loading: the weights being read from disk and arranged in memory. The second, shorter pause is generation itself. On a CPU with the 10M parameter model used in this chapter, generation takes a fraction of a second per response. On a GPU with a 70B parameter model, it takes several seconds — which is exactly why throughput optimisation matters and why this book exists.

Run the program a second time without changing anything. The output will be different. That is sampling: the model is not computing a single deterministic answer, it is drawing from a probability distribution, and that distribution has many plausible continuations. Whether that nondeterminism is a feature or a bug depends on the application, and controlling it precisely is what the sampling parameters are for.

## What comes next

Each subsequent section in this chapter takes one part of the program above and opens it. `01_load.py` shows what `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` actually do — what is downloaded, how the weights are laid out in memory, and how long loading takes as model size grows. `02_sampling.py` shows what happens when you change `temperature`, `top_k`, and `top_p`, running the same prompt through each configuration so you can see the difference directly. `03_contracts.py` introduces the shared data types that every chapter in this book uses, so that when we replace the engine in later chapters the surrounding code stays the same.

By the time you reach the server and the benchmark, you will have seen every component of this system working in isolation. Then, when Chapter 2 starts changing things, you will know exactly which box is being opened and why.
