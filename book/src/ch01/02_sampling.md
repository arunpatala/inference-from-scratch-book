# Sampling

Every time a language model generates a token it produces a vector of raw scores — one per word in its vocabulary — and then has to choose which word to output next. How it makes that choice has an enormous impact on the character of the text it produces. The same model, given the same prompt, can output a careful and predictable answer, a varied and creative one, or something approaching nonsense, depending entirely on the sampling strategy applied to those scores. This section makes that choice concrete and observable.

```python
def generate(model, tokenizer, prompt: str, max_new_tokens: int = BENCHMARK_TOKENS, **kwargs):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)
    new_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
```

Run it:

```
python 02_sampling.py
python 02_sampling.py --prompt "Once upon a time"
```

The `generate` helper above strips the prompt tokens from the output before decoding — `output_ids[0][input_ids.shape[1]:]` selects only the newly generated tokens — so what you see printed is the model's continuation, not a repetition of the input. The `**kwargs` pattern passes whatever sampling arguments the caller provides directly to `model.generate`, which is what allows the demo below to explore every strategy through the same function.

## From logits to a probability distribution

Before any sampling strategy makes sense, it helps to understand what the model actually produces at each decode step. The final layer of the network outputs a vector of floating-point numbers called logits — one number for each token in the vocabulary, which for most modern models means somewhere between 32,000 and 128,000 values. A logit is a raw score: a higher number means the model considers that token a more plausible continuation of the sequence so far. The numbers themselves are not probabilities; they can be negative, arbitrarily large, and they do not sum to one.

To turn logits into probabilities the model applies the softmax function, which exponentiates each logit and then divides by the sum of all exponentiated logits. The result is a proper probability distribution: every value is positive and they sum to exactly one. A token with a logit of 3.5 and one with a logit of 3.0 will not have probabilities of 3.5% and 3.0% — the exponential function amplifies differences, so the gap between them in probability space is larger than the gap in logit space.

Once the probabilities are computed, some form of sampling algorithm picks the next token. The sampling parameters — `temperature`, `top_k`, `top_p` — all operate on the logits before or alongside the softmax step, reshaping the distribution before a token is drawn from it. Chapter 3 will implement this from scratch; for now the goal is to observe the effects.

## Greedy decoding

```python
{"do_sample": False}
```

The simplest strategy is to always pick the token with the highest probability. This is called greedy decoding, and it is what `do_sample=False` selects. The demo runs the same prompt twice with identical settings.

```python
{"label": "Greedy / temperature=0  [run 1]", "kwargs": {"do_sample": False}},
{"label": "Greedy / temperature=0  [run 2]", "kwargs": {"do_sample": False}},
```

Both runs produce identical output. That is the defining property of greedy decoding: given the same prompt and the same model weights, the output is completely deterministic. There is no randomness anywhere in the computation. This makes greedy decoding reproducible and easy to debug, which is why it is the default for tasks like translation or summarisation where there is typically one correct answer. Its weakness is that it tends toward repetitive, formulaic text — the most probable token at each step is often also the most expected one, and the model can get trapped following its own highest-probability paths into loops.

## Temperature

```python
{"do_sample": True, "temperature": 0.1},
{"do_sample": True, "temperature": 1.0},
{"do_sample": True, "temperature": 2.0},
```

Temperature is a scalar that divides every logit before the softmax is applied. The modified formula is:

\\[ p_i = \frac{e^{x_i / T}}{\sum_{j} e^{x_j / T}} \\]

where \\(T\\) is the temperature and \\(x_i\\) is the raw logit for token \\(i\\). When \\(T < 1\\), dividing by a number less than one makes each logit larger in magnitude. Because the softmax uses exponentials, this amplifies the differences between tokens: already-probable tokens become even more dominant, and unlikely tokens become nearly impossible. The output concentrates around the model's top choices and becomes more predictable — and more prone to repetition. When \\(T > 1\\), the logits shrink toward zero, softmax differences flatten, and the probability mass spreads more evenly across the vocabulary. Lower-ranked tokens get a larger share of the probability, and the output becomes more varied — at the cost of sometimes choosing tokens that are implausible in context.

At `temperature=0` the difference between logits becomes infinite in the limit, and the distribution collapses to a point mass on the single highest-probability token — equivalent to greedy decoding. At very high temperatures the distribution approaches uniform, meaning every token in the vocabulary has roughly equal probability regardless of context.

Running the demo shows this directly. At `temperature=0.1` the two runs with the same prompt produce nearly identical output — the distribution is so concentrated that the same token wins almost every time. At `temperature=1.0` the two runs diverge. At `temperature=2.0` the output may be surprising, non sequitur, or grammatically strange, because the model is now sampling from a flattened distribution that gives real probability to tokens it would ordinarily never choose.

## Top-k sampling

```python
{"do_sample": True, "temperature": 1.0, "top_k": 1},
{"do_sample": True, "temperature": 1.0, "top_k": 5},
{"do_sample": True, "temperature": 1.0, "top_k": 200},
```

Top-k sampling addresses a specific problem with pure temperature sampling: even at moderate temperatures, the long tail of the vocabulary — tokens with very low probability — retains some nonzero probability mass, and occasionally one of them gets sampled. Most of those tokens are genuinely bad choices that no amount of creativity justifies.

Top-k solves this by discarding all but the \\(k\\) most probable tokens before sampling. The logits of the remaining vocabulary are set to negative infinity, which makes their softmax probability exactly zero. Sampling then proceeds over just the \\(k\\) survivors. When \\(k = 1\\), only the single best token survives and the result is identical to greedy decoding. When \\(k\\) equals the full vocabulary size, nothing is filtered and the result is identical to unconstrained sampling. Values between those extremes offer a tunable trade-off.

The weakness of top-k is that it uses a fixed count regardless of the shape of the distribution. At a position where the model is highly confident — the next token is almost certainly "the" — a top-k of 50 still allows 49 other tokens to compete, including plausible but wrong ones. At a position where the model is genuinely uncertain — many continuations are reasonable — a top-k of 5 may be too restrictive. The right value of \\(k\\) is context-dependent in a way the parameter cannot capture.

## Top-p (nucleus) sampling

```python
{"do_sample": True, "temperature": 1.0, "top_p": 0.1},
{"do_sample": True, "temperature": 1.0, "top_p": 0.95},
```

Top-p sampling, introduced by Holtzman et al. in 2019, solves the rigidity of top-k by filtering based on cumulative probability mass rather than a fixed count. Tokens are ranked from most to least probable, and the model keeps adding tokens to the candidate set until their cumulative probability exceeds the threshold \\(p\\). All remaining tokens are set to zero probability. Sampling then proceeds over the nucleus — the smallest set of tokens that together account for at least \\(p\\) of the probability mass.

The result is a candidate set that adapts to the situation. When the model is confident, probability mass is concentrated in a few tokens and the nucleus is small — perhaps two or three tokens — even if \\(p = 0.95\\). When the model is uncertain, the mass is spread across many tokens and the nucleus expands accordingly. The number of candidates is not fixed; it is determined by the shape of the distribution at each step.

In the demo, `top_p=0.1` produces very conservative output: only the tokens that together account for the first 10% of probability mass are considered, which typically means just one or two tokens. `top_p=0.95` allows a much richer set of candidates. Most production systems use values between 0.9 and 0.95 as a reasonable default.

## What to observe when you run it

The most informative thing to look for is the greedy runs first. Their identical outputs confirm that when sampling is disabled the computation is purely deterministic. Then compare the two `temperature=1.0` runs — they should differ, sometimes substantially. That difference comes entirely from the randomness introduced by sampling, not from anything in the model itself.

The comparison between `top_k=1` and `do_sample=False` is also instructive: they produce identical output, which confirms that restricting the candidate pool to a single token is the same as always taking the argmax. And the comparison between `top_p=0.1` and `top_p=0.95` shows nucleus sampling adapting to different thresholds — the conservative setting staying close to the model's top choices, the permissive setting occasionally venturing further.

None of the math underlying these strategies has been implemented here — `model.generate` handles all of it. Section 1.4 introduces the shared `SamplingParams` dataclass that every subsequent chapter uses to carry these parameters through the system. Chapter 3 implements the sampling logic from scratch, making the formulas above executable and inspectable.
