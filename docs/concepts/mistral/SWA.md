# Sliding Window Attention (SWA)

## 1. The Problem: Full Attention Grows Quadratically

In standard causal attention, every token attends to every previous token:

$$\text{score}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}$$

For a sequence of length $T$, the attention matrix is $T \times T$. The compute cost is $O(T^2)$ and the KV cache memory is $O(T)$.

During decode generation:

- Step 1: attend to 1 token
- Step 2: attend to 2 tokens
- Step N: attend to N tokens

This means both compute and memory grow continuously with sequence length. For a 32K context window, this becomes a serious bottleneck.

---

## 2. SWA's Key Insight

Most information that a token needs is **local** — nearby tokens in the sequence.

SWA restricts each token to attend only to a fixed-size window of $W$ most recent tokens rather than all previous tokens:

$$\text{token at position } i \text{ attends to positions } \max(0, i - W)\ \text{ to }\ i$$

This caps both compute and active KV cache size at $O(W)$ regardless of total sequence length.

For Mistral 7B v0.1: $W = 4096$.

---

## 3. How SWA Works — Step by Step

### Step 1: Normal Q, K, V projection

Nothing changes in the projections. GQA still applies.

### Step 2: KV cache update

New K, V tokens are appended to the cache as normal (unrotated in our implementation).

### Step 3: Select active KV window

Instead of reading the full cache history, we select only the most recent $W$ tokens:

$$k_{\text{active}} = k[..., \max(0, T_{\text{total}} - W) : T_{\text{total}}, :]$$

The active window has length $W_{\text{active}} = \min(T_{\text{total}}, W)$.

### Step 4: Apply RoPE with correct absolute positions

The active window starts at absolute position $k_{\text{start}} = \max(0, T_{\text{total}} - W)$.

$$q = \text{RoPE}(q,\ \text{start\_pos} = T_{\text{total}} - T)$$
$$k_{\text{active}} = \text{RoPE}(k_{\text{active}},\ \text{start\_pos} = k_{\text{start}})$$

### Step 5: Build windowed causal mask

Only allow each query token to attend to keys within the causal window:

```bash
token at position i can see: max(0, i - W) ... i
```

Mask shape: `(T, W_active)` instead of `(T, T_total)`.

### Step 6: Standard attention computation

$$\text{score} = q \cdot k_{\text{active}}^T \cdot \text{scale}$$
$$\text{attn} = \text{softmax}(\text{score} + \text{mask}) \cdot v_{\text{active}}$$

---

## 4. Sliding Window vs Full Causal: Visual Comparison

Full causal attention (T=6):

```bash
     k0  k1  k2  k3  k4  k5
q0 [  1   0   0   0   0   0 ]
q1 [  1   1   0   0   0   0 ]
q2 [  1   1   1   0   0   0 ]
q3 [  1   1   1   1   0   0 ]
q4 [  1   1   1   1   1   0 ]
q5 [  1   1   1   1   1   1 ]
```

Sliding window attention (W=3, T=6):

```bash
     k0  k1  k2  k3  k4  k5
q0 [  1   0   0   0   0   0 ]
q1 [  1   1   0   0   0   0 ]
q2 [  1   1   1   0   0   0 ]
q3 [  0   1   1   1   0   0 ]
q4 [  0   0   1   1   1   0 ]
q5 [  0   0   0   1   1   1 ]
```

Each query can see at most $W$ past tokens. Older tokens are invisible once they exit the window.

---

## 5. Impact on Long Sequences

| Property             | Full Attention   | SWA               |
|----------------------|------------------|-------------------|
| Attention compute    | $O(T^2)$         | $O(T \cdot W)$    |
| Active KV per step   | $O(T)$           | $O(W)$            |
| RoPE per decode step | $O(T \cdot d)$   | $O(W \cdot d)$    |
| Long-context quality | Full context     | Local context only|

Where $W$ is the window size and $d$ is head dimension.

---

## 6. Mistral Version Comparison

| Version | SWA | Window Size | Context | Notes                            |
|---------|-----|-------------|---------|----------------------------------|
| v0.1    | Yes | 4096        | 8K      | Original SWA design              |
| v0.2    | No  | —           | 32K     | Full attention, expanded context |
| v0.3    | No  | —           | 32K     | Same as v0.2, extended vocab     |

Config field: `sliding_window`. If `null` or missing, use full causal attention.

---

## 7. SWA and KV Cache Interaction

**Important:** SWA does not require storing only the window in cache. The combination with GQA, paged cache, and serving requires care:

| Design             | What cache stores | What attention reads | Notes                          |
|--------------------|-------------------|----------------------|--------------------------------|
| Full store, W read | Full history      | Active window only   | Our approach — preserves reuse |
| Store W, W read    | Only last W       | Full stored cache    | Simpler but loses prefix reuse |

Our implementation stores full unrotated KV, but attention reads only the active window. This preserves paged block reuse across requests while bounding compute.

---

## 8. Interview-Ready Points

### Q: "Why does SWA not hurt quality much despite missing old tokens?"

Most syntactic and semantic dependencies are local. Long-range dependencies in text are rarer than local ones. Models trained with SWA learn to encode sufficient information in nearby context. Additionally, deep stacking of SWA layers provides an effective receptive field that grows with depth.

### Q: "What is the effective receptive field with SWA?"

With window size $W$ and $L$ layers, the receptive field grows as $O(W \cdot L)$.
For Mistral: $W = 4096$, $L = 32$, so effective field covers approximately 131K tokens of indirect influence across layers. This is why SWA models can handle longer contexts than the raw window size suggests.

### Q: "Does SWA affect the KV cache format?"

No, in our design. KV cache stores unrotated full history. SWA only affects which slice of the cache is read and which positions are rotated at attention time.

### Q: "How do you handle the mask with SWA?"

Instead of a full $T_{\text{total}} \times T_{\text{total}}$ lower-triangular mask, you build a smaller $(T, W_{\text{active}})$ mask where each row allows only tokens within the causal window. This also avoids the expensive full mask allocation.

### Q: "How does k start_pos change with SWA?"

With full causal attention, cache returns full history so `k start_pos = 0`. With SWA, cache returns only the active window starting at absolute position $\max(0, T_{\text{total}} - W)$, so `k start_pos = max(0, T_total - W)`. This is the key position bookkeeping change SWA introduces.

---

## 9. Implementation Scope

**File:** `src/llm_engine/model/Mistral/attention.py`

**Current state:** Full causal attention is implemented. SWA is planned as a config-driven extension.

**What is implemented:**

`MistralAttention.forward()` — full causal path:

- Builds full lower-triangular causal mask over `T_total` tokens
- Applies RoPE to `q` with `start_pos = T_total - T` and to `k` with `start_pos = 0`
- Attends over the full returned cache history

**What is planned — SWA extension:**

`sliding_window` parameter added to `__init__`:

- Read from `config.json` field `sliding_window`
- If `None` or not set: use full causal attention (current behavior)
- If positive integer: use windowed causal attention

`forward()` changes when `sliding_window` is active:

- Slice active KV window: `k[..., max(0, T_total - W):T_total, :]`
- Apply RoPE to `k` with `start_pos = max(0, T_total - W)` instead of `0`
- Build smaller windowed mask of shape `(T, W_active)` instead of `(T, T_total)`
- Attend only over the active window, not full history

**Called from:** Each `MistralBlock` layer in the full model forward pass, passing `layer_idx` and `kv_cache`.
