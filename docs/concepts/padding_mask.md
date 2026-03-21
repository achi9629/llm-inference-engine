# Padding Mask: Design, Bugs, and Fixes

## Why Padding Is Needed

Batch inference requires all sequences to have the same length. Shorter sequences must be padded. For causal (autoregressive) LMs like GPT-2, **left padding** is used — pad tokens go on the left so the last token position always reflects full real context and produces valid next-token logits.

```bash
Sequence A: [1, 2, 3]        → left padded: [PAD, PAD, 1, 2, 3]
Sequence B: [6, 7, 8, 9, 10] → no padding:  [6,   7,   8, 9, 10]
```

## Padding Mask Convention

- `1` = real token
- `0` = pad token

This matches HuggingFace's `attention_mask` convention.

```bash
Sequence A mask: [0, 0, 1, 1, 1]
Sequence B mask: [1, 1, 1, 1, 1]
```

## Pipeline: How Padding Mask Flows

1. **Tokenizer** (`tokenizer.py`): `batch_encode()` returns `(input_ids, attention_mask)` with `padding_side="left"`
2. **InferenceEngine** (`inference_engine.py`): Unpacks `token_ids, padding_mask = self.encode(input_text)`
3. **Generator** (`generator.py`): Passes `padding_mask` to model. Appends `1` for each new generated token
4. **Transformer** (`transformer.py`): Computes position IDs from padding mask, passes mask to each block
5. **Attention** (`attention.py`): Masks out pad positions in attention scores

## Problem 1: NaN from `float("-inf")` Masking

### Root Cause for Problem 1

Left-padded positions (e.g., positions 0 and 1 for mask `[0, 0, 1, 1, 1]`) can only attend to positions ≤ themselves (causal mask). But those positions are ALL pad tokens (padding mask = 0). With `float("-inf")` masking:

```bash
Position 0 attention scores → all -inf → softmax([-inf]) = NaN
```

NaN propagates through residual connections and LayerNorm, corrupting the entire sequence — including real token positions.

### Fix: Use `torch.finfo(score.dtype).min` Instead of `float("-inf")`

```python
# Before (broken):
score = score.masked_fill(causal_mask == 0, float("-inf"))
score = score.masked_fill(padding_mask == 0, float("-inf"))

# After (fixed):
score = score.masked_fill(causal_mask == 0, torch.finfo(score.dtype).min)
score = score.masked_fill(padding_mask == 0, torch.finfo(score.dtype).min)
```

**Why it works:** `torch.finfo(score.dtype).min` returns the most negative **finite** value for the tensor's dtype. `exp(very_large_negative) ≈ 0`, so softmax produces near-zero (not NaN) for all-masked rows. The residual connection then passes the original embedding through unchanged — correct behavior for pad positions.

**Dtype values:**

| dtype    | `finfo.min`     |
|----------|-----------------|
| float32  | `-3.4028e+38`   |
| float16  | `-65504.0`      |
| bfloat16 | `-3.3895e+38`   |

This is dtype-aware and works for all common precisions.

### Why Not `nan_to_num(0.0)` After Softmax?

That's a reactive patch — lets NaN happen, then cleans up. Using `finfo.min` prevents NaN from occurring. It also avoids silently masking real NaN bugs elsewhere.

## Problem 2: Wrong Position IDs with Left Padding

### Root Cause for Problem 2

Original position encoding used a simple range:

```python
pos_ids = torch.arange(start_pos, start_pos + T).unsqueeze(0)
# For [PAD, PAD, 1, 2, 3] → positions [0, 1, 2, 3, 4]
```

This gives pad tokens real positions, shifting all real tokens to higher positions than they should have. A single unpacked `[1, 2, 3]` gets positions `[0, 1, 2]`, but the same tokens in a padded batch get positions `[2, 3, 4]` — different position embeddings → different outputs.

### Fix: Cumsum-Based Position IDs

```python
if padding_mask is not None:
    pos_ids = (padding_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    if T == 1:  # decode step with KV cache
        pos_ids = pos_ids[:, -1:]
else:
    pos_ids = torch.arange(start_pos, start_pos + T, device=input_ids.device).unsqueeze(0)
```

**Example:**

```bash
mask:    [0, 0, 1, 1, 1]
cumsum:  [0, 0, 1, 2, 3]
- 1:     [-1,-1, 0, 1, 2]
clamp:   [0, 0, 0, 1, 2]  ← pad positions get pos 0, real tokens get 0,1,2
```

Real tokens `[1, 2, 3]` now get positions `[0, 1, 2]` — identical to the unpadded case.

## Verification: `test_padding_doesnt_corrupt`

The definitive test that proves correctness:

```python
# Single sequence [1, 2, 3] — no padding
output_single = generator(model, token_ids=[1,2,3], padding_mask=None, ...)

# Same sequence in a padded batch [PAD, PAD, 1, 2, 3]
output_batch = generator(model, token_ids=[[PAD,PAD,1,2,3], ...],
                         padding_mask=[[0,0,1,1,1], ...], ...)

# Strip padding prefix and compare
assert output_single == output_batch[0, n_pad:]  # ✅ Exact match
```

Generated tokens are identical — padding has zero effect on prediction.
