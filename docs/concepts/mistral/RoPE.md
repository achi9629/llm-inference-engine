# RoPE (Rotary Position Embeddings)

## 1. The Problem: Why Transformers Need Position Information

Self-attention is **permutation-invariant** — `Attention(Q, K, V)` produces the same output regardless of token order. Without position information, "the cat sat on the mat" and "mat the on sat cat the" look identical to the model.

**GPT-2's solution — Learned absolute position embeddings:**

```python
x = token_embedding + position_embedding[pos]
```

A learned `nn.Embedding(max_seq_len, d_model)` lookup. Each position index maps to a unique learned vector — position 0 always retrieves the same $p_0$, position 1 always retrieves $p_1$, etc. These are **absolute** positions: the vector for position 5 is fixed regardless of what tokens surround it.

**Problems with this approach:**

- **Fixed max length** — cannot extrapolate beyond `max_seq_len` seen during training
- **No relative awareness** — the model must learn from data that position 5 is "2 tokens after" position 3. There's no structural inductive bias for relative distance.
- **Position info decays** — the position signal is added at the input and must survive through all layers. By deeper layers, position information may be diluted.

---

## 2. RoPE's Key Insight

Instead of adding position information to the **input embeddings**, RoPE encodes position **directly into Q and K** before the attention dot product.

The core property RoPE achieves:

$$q_m^\top k_n = f(q, k, m - n)$$

The dot product between query at position $m$ and key at position $n$ depends only on the **relative distance** $(m - n)$, not on absolute positions $m$ and $n$ individually.

**Why this matters:**

- Token at position 100 attending to position 98 behaves the same as position 50 attending to position 48 — both are "2 tokens back"
- The model doesn't need to waste capacity learning absolute position patterns
- Better length generalization — relative patterns transfer to unseen sequence lengths

---

## 3. How RoPE Works — Step by Step

### Step 1: Pair up dimensions

Given a head dimension $d$, group consecutive dimensions into $d/2$ pairs: $(x_0, x_1), (x_2, x_3), \ldots, (x_{d-2}, x_{d-1})$.

Each pair is treated as a 2D vector that gets **rotated** by a position-dependent angle.

Here, $m$ denotes a token's position in the sequence: $m = 0$ for the first token, $m = 1$ for the second, up to $m = T-1$ for the last token in a prompt of length $T$.

### Step 2: Compute rotation frequencies

Each pair $i$ gets a base frequency:

$$\theta_i = \frac{1}{10000^{2i/d}}, \quad i = 0, 1, \ldots, d/2 - 1$$

- Pair 0: highest frequency (rotates fast) — captures fine-grained local position
- Pair $d/2 - 1$: lowest frequency (rotates slow) — captures long-range position

At position $m$, pair $i$ rotates by angle $m \cdot \theta_i$.

### Step 3: Apply rotation

For each pair $(x_{2i}, x_{2i+1})$ at position $m$:

$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

This is a standard 2D rotation matrix. Each dimension pair lives in its own rotation plane.

### Step 4: Why the dot product becomes relative

After rotating Q at position $m$ and K at position $n$:

$$q_m^\top k_n = \sum_i \left[ q_{2i} k_{2i} \cos((m-n)\theta_i) - q_{2i} k_{2i+1} \sin((m-n)\theta_i) + \ldots \right]$$

The rotation angles subtract: $m\theta_i - n\theta_i = (m-n)\theta_i$. The absolute positions cancel, leaving only the relative distance.

---

## 4. Efficient Implementation (Complex Number Trick)

Rather than materializing $d/2$ rotation matrices, the standard implementation uses **complex multiplication**:

1. **View as complex:** reshape $(B, T, H, d)$ → $(B, T, H, d/2, 2)$ → view as complex $(B, T, H, d/2)$
2. **Precompute frequencies:** $e^{im\theta_i} = \cos(m\theta_i) + i\sin(m\theta_i)$ for all positions $m$ and pairs $i$
3. **Element-wise multiply:** `x_complex * freqs_complex` — this IS the rotation
4. **View back as real:** convert complex → real, reshape back to $(B, T, H, d)$

Complex multiplication $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$ is exactly the 2D rotation matrix applied to $(a, b)$ by angle $\theta$ where $c = \cos\theta, d = \sin\theta$.

**Precomputation:** The frequency tensor `freqs_cis` of shape `(max_seq_len, d/2)` is computed **once** at init and cached. During forward, we just slice `freqs_cis[start_pos : start_pos + T]` — no recomputation per step.

---

## 5. RoPE vs Other Position Encodings

| Method                   | Type         | Relative Aware | Length Extrapolation | Where Applied    | Parameters          |
|--------------------------|--------------|----------------|----------------------|------------------|---------------------|
| Sinusoidal (Transformer) | Fixed        | No             | Poor                 | Input embedding  | 0                   |
| Learned absolute (GPT-2) | Learned      | No             | None (hard max)      | Input embedding  | `max_len × d_model` |
| ALiBi (BLOOM, Falcon)    | Fixed bias   | Yes            | Good                 | Attention scores | 0                   |
| **RoPE (Mistral)**       | **Rotation** | **Yes**        | **Moderate**         | **Q and K only** | **0**               |

**Key differences from GPT-2:**

- **Zero learnable parameters** — frequencies are deterministic from the formula
- **Applied to Q/K, not input** — position info is fresh at every layer, doesn't decay
- **Only affects attention** — V vectors and FFN are position-free, which is cleaner
- **Relative by construction** — the math guarantees relative distance awareness

---

## 6. Interview-Ready Points

### Q: "Why does Mistral use RoPE instead of learned position embeddings?"

> Three reasons: (1) RoPE encodes **relative** position by construction — the Q·K dot product depends on $(m - n)$, not on absolute positions individually. Learned embeddings encode absolute positions and must learn relative patterns from data. (2) RoPE has **zero learnable parameters** — positions are encoded via deterministic rotation frequencies. (3) RoPE is applied **directly to Q/K at every layer**, so position information doesn't decay through the network like additive embeddings do.

### Q: "How does the rotation encode position?"

> Head dimensions are grouped into pairs. Each pair is treated as a 2D vector and rotated by a position-dependent angle. Different pairs rotate at different frequencies — fast frequencies capture local position, slow frequencies capture long-range position. When Q at position $m$ and K at position $n$ are dot-producted, the rotation angles subtract, so the result depends only on the relative distance $m - n$.

### Q: "What's the `10000` base in the frequency formula?"

> It's $\theta_{\text{base}}$ — controls the wavelength range. $\theta_i = 1/10000^{2i/d}$ creates a geometric progression from high frequency (pair 0, wavelength ~$2\pi$) to low frequency (last pair, wavelength ~$2\pi \cdot 10000$). Higher base = longer wavelengths = better long-context extrapolation. Mistral uses `rope_theta=500000` instead of `10000` for this reason.

### Q: "How does it work with KV cache during decode?"

> During prefill, all positions $[0, 1, \ldots, T-1]$ are rotated at once. During decode, only the single new token at position $T$ needs rotation. The key insight is that **cached K vectors are already rotated** at their original positions — we don't re-rotate them. We only rotate the new Q (at current position) and the new K (at current position), then the Q·K dot product naturally produces relative distances against all cached positions.

---

## 7. Implementation Scope

**File:** `src/llm_engine/model/Mistral/rope.py`

**Two functions:**

`precompute_freqs_cis(dim, max_seq_len, theta=10000.0)`:

- Compute $\theta_i = 1/\text{theta}^{2i/\text{dim}}$ for $i = 0 \ldots \text{dim}/2 - 1$
- Compute outer product: `positions × freqs` → shape `(max_seq_len, dim/2)`
- Return as complex: $e^{im\theta_i}$ → `torch.polar(ones, angles)` → shape `(max_seq_len, dim/2)`

`apply_rotary_emb(x, freqs_cis)`:

- Reshape x: `(B, T, H, d)` → `(B, T, H, d/2, 2)` → view as complex `(B, T, H, d/2)`
- Slice freqs_cis to match T
- Multiply: `x_complex * freqs_cis` (broadcasts over B and H)
- View back as real → reshape to `(B, T, H, d)`

**Called in attention:** `q = apply_rotary_emb(q, freqs_cis)`, `k = apply_rotary_emb(k, freqs_cis)` — before the Q·K dot product, after the Q/K projection.
