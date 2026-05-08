# GQA (Grouped Query Attention)

## 1. The Problem: KV Cache is the Bottleneck

**GPT-2's approach — Multi-Head Attention (MHA):**

```python
# Each head has its own Q, K, V
Q = x @ W_q  # (B, T, n_heads, head_dim)
K = x @ W_k  # (B, T, n_heads, head_dim)
V = x @ W_v  # (B, T, n_heads, head_dim)
```

Every attention head has its own query, key, and value projections. During autoregressive decoding, K and V tensors from all previous tokens must be cached (the KV cache) to avoid recomputation.

**The KV cache memory problem:**

For a model with $H$ heads, head dimension $d_h$, sequence length $T$, $L$ layers, and batch size $B$:

$$\text{KV cache} = 2 \times B \times L \times H \times T \times d_h \times \text{bytes\_per\_param}$$

For Mistral 7B ($H=32, d_h=128, L=32$) with batch size 16, sequence length 2048, in float16:

$$2 \times 16 \times 32 \times 32 \times 2048 \times 128 \times 2 = 17.2 \text{ GB}$$

That's **17 GB just for the KV cache** — a significant fraction of GPU memory that limits batch size and therefore throughput.

**Problems with MHA's KV cache:**

- **Memory scales linearly with n_heads** — each head stores its own K and V, so 32 heads = 32x the per-head KV memory
- **Limits batch size** — more KV cache memory = fewer concurrent sequences the GPU can handle
- **Memory bandwidth bound** — during decode, loading KV cache from HBM is the bottleneck, not compute. More KV heads = more bytes to load per step
- **Throughput ceiling** — serving systems hit OOM before the GPU compute is saturated

---

## 2. Multi-Query Attention (MQA): The Extreme Solution

MQA (Noam Shazeer, 2019) takes a radical approach: **all query heads share a single K and a single V**.

```python
Q = x @ W_q  # (B, T, n_heads, head_dim)    — 32 heads
K = x @ W_k  # (B, T, 1, head_dim)           — 1 KV head
V = x @ W_v  # (B, T, 1, head_dim)           — 1 KV head
```

The single K and V are broadcast across all query heads during the attention computation.

**KV cache with MQA (Mistral 7B dimensions, same scenario):**

$$2 \times 16 \times 32 \times 1 \times 2048 \times 128 \times 2 = 0.54 \text{ GB}$$

**32x reduction** in KV cache memory — from 17 GB to 0.5 GB.

**Benefits of MQA:**

- Dramatically smaller KV cache → larger batch sizes → higher throughput
- Less memory bandwidth needed during decode → lower latency per token
- Fewer KV parameters → faster attention computation

**Problems with MQA:**

- **Quality degradation** — a single KV representation is too compressed. Different query heads need different "views" of the key-value information. Forcing all 32 heads to attend to the same K/V hurts model quality, especially on complex reasoning tasks.
- **All-or-nothing** — no middle ground between 1 KV head (MQA) and 32 KV heads (MHA). You either get the full memory savings with quality loss, or full quality with the memory problem.

---

## 3. GQA's Key Insight

Grouped Query Attention (Ainslie et al., 2023) finds the **middle ground**: instead of 1 KV head (MQA) or $H$ KV heads (MHA), use $G$ KV head groups where $1 < G < H$.

$$n_{\text{kv\_heads}} = G, \quad \text{group size} = H / G$$

Each group of query heads shares one K head and one V head.

For Mistral 7B: $H = 32$ query heads, $G = 32$ KV heads (actually MHA).
For LLaMA 70B: $H = 64$ query heads, $G = 8$ KV heads → 8 groups of 8 query heads each.

**Why this matters:**

- **Tunable tradeoff** — choose $G$ to balance quality vs memory. More KV groups = closer to MHA quality. Fewer = closer to MQA efficiency.
- **Near-MHA quality** — experiments show that even $G = 8$ (8x KV cache reduction) recovers almost all MHA quality.
- **Near-MQA speed** — the KV cache shrinks by $H/G$, giving most of MQA's throughput benefit.
- **Simple implementation** — just repeat (broadcast) each KV head to serve its group of query heads. One extra operation: `repeat_kv`.

---

## 4. How GQA Works — Step by Step

### Step 1: Separate Q and KV projections

Unlike MHA where Q, K, V all have `n_heads` heads:

- $W_Q \in \mathbb{R}^{d_{\text{model}} \times (n_{\text{heads}} \times d_h)}$ — full query heads
- $W_K \in \mathbb{R}^{d_{\text{model}} \times (n_{\text{kv\_heads}} \times d_h)}$ — fewer KV heads
- $W_V \in \mathbb{R}^{d_{\text{model}} \times (n_{\text{kv\_heads}} \times d_h)}$ — fewer KV heads

### Step 2: Project Q, K, V

```python
Q = x @ W_q  →  (B, T, n_heads, head_dim)       # e.g., 64 heads
K = x @ W_k  →  (B, T, n_kv_heads, head_dim)     # e.g., 8 heads
V = x @ W_v  →  (B, T, n_kv_heads, head_dim)     # e.g., 8 heads
```

### Step 3: Apply RoPE to Q and K

```python
Q = apply_rotary_pos_emb(Q)  # position encoding on all 64 Q heads
K = apply_rotary_pos_emb(K)  # position encoding on all 8 K heads
```

### Step 4: Expand KV heads to match Q heads (`repeat_kv`)

Each KV head is repeated `n_heads // n_kv_heads` times:

```python
# K: (B, T, 8, head_dim) → (B, T, 64, head_dim)
# Each of the 8 KV heads is copied 8 times
K = repeat_kv(K, n_rep=n_heads // n_kv_heads)
V = repeat_kv(V, n_rep=n_heads // n_kv_heads)
```

After expansion, K and V have the same number of heads as Q, so standard attention math works unchanged.

### Step 5: Standard attention computation

$$\text{score} = \frac{Q K^\top}{\sqrt{d_h}} \quad \rightarrow \quad \text{softmax} \quad \rightarrow \quad \text{score} \times V$$

From here, it's identical to MHA — causal mask, softmax, weighted sum.

---

## 5. The `repeat_kv` Operation

This is the only new operation GQA introduces. It copies each KV head to serve its group:

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, T, n_kv_heads, head_dim) → (B, T, n_heads, head_dim)"""
    if n_rep == 1:
        return x  # MHA: no repetition needed
    B, T, n_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(B, T, n_kv_heads, n_rep, head_dim)
    return x.reshape(B, T, n_kv_heads * n_rep, head_dim)
```

**What happens:** `(B, T, 8, head_dim)` → insert dim → `(B, T, 8, 1, head_dim)` → expand → `(B, T, 8, 8, head_dim)` → reshape → `(B, T, 64, head_dim)`.

**Memory note:** `expand` doesn't copy data — it creates a view with stride 0. The actual copy happens at `reshape`. During training, this is fine. During inference with KV cache, the repetition happens on the fly and the cached tensors stay at `n_kv_heads` size.

---

## 6. MHA vs MQA vs GQA Comparison

| Property           | MHA          | MQA                        | GQA                           |
|--------------------|--------------|----------------------------|-------------------------------|
| Q heads            | $H$          | $H$                        | $H$                           |
| KV heads           | $H$          | 1                          | $G$ (where $1 < G < H$)       |
| KV cache size      | $2BLHTd_h$   | $2BLTd_h$                  | $2BLGTd_h$                    |
| KV cache reduction | 1x           | $H$x                       | $H/G$x                        |
| Quality            | Best         | Degraded                   | Near-MHA                      |
| Throughput         | Lowest       | Highest                    | Near-MQA                      |
| Used in            | GPT-2, GPT-3 | PaLM (original), Falcon-7B | **LLaMA 2/3, Mistral, Gemma** |

**Mistral / LLaMA / Falcon model configurations:**

| Model       | n_heads | n_kv_heads | Group size | KV reduction |
|-------------|---------|------------|------------|--------------|
| LLaMA 7B    | 32      | 32         | 1 (MHA)    | 1x           |
| LLaMA 13B   | 40      | 40         | 1 (MHA)    | 1x           |
| LLaMA 70B   | 64      | 8          | 8          | 8x           |
| LLaMA 3 8B  | 32      | 8          | 4          | 4x           |
| Mistral 7B  | 32      | 8          | 4          | 4x           |
| Falcon-7B   | 71      | 1          | 71 (MQA)   | 71x          |
| Falcon-40B  | 128     | 8          | 16         | 16x          |
| Falcon-180B | 232     | 8          | 29         | 29x          |

Note: LLaMA 7B/13B use full MHA (n_kv_heads = n_heads). Mistral 7B and LLaMA 3 8B use GQA with 8 KV heads. Falcon-7B uses extreme MQA (1 KV head for 71 query heads). GQA is preferred when KV cache becomes a real bottleneck.

---

## 7. Interview-Ready Points

### Q: "What's the difference between MHA, MQA, and GQA?"

> MHA gives every query head its own K and V — best quality but KV cache scales with n_heads. MQA shares a single K and V across all query heads — 32x KV cache reduction but quality drops. GQA is the middle ground: n_kv_heads groups, each serving n_heads/n_kv_heads query heads. LLaMA 70B uses 8 KV heads for 64 query heads — 8x KV cache reduction with near-MHA quality.

### Q: "Why does KV cache matter more than QKV projection cost?"

> During autoregressive decode, each step generates one token. The compute is tiny (one token x model), but the KV cache for all previous tokens must be loaded from HBM. With long sequences and large batches, this memory bandwidth is the bottleneck — not FLOPs. Reducing KV heads directly reduces the bytes loaded per decode step, which is why MQA/GQA improve inference throughput even though they barely change training FLOPs.

### Q: "How does `repeat_kv` work and is it expensive?"

> It copies each KV head n_rep times to match the number of query heads, so standard attention math works unchanged. It uses `expand` (a stride-0 view, no memory copy) followed by `reshape` (contiguous copy). The cost is small compared to the attention matmul itself. Some implementations use `torch.repeat_interleave` instead, but expand+reshape is more explicit and avoids kernel launch overhead.

### Q: "How does GQA affect the KV cache implementation?"

> The cache stores tensors of shape `(B, n_kv_heads, T, head_dim)` instead of `(B, n_heads, T, head_dim)`. When n_kv_heads < n_heads, the cache is proportionally smaller. The `repeat_kv` expansion happens **after** reading from cache, right before computing attention scores. So cached K/V stay compact; only the working tensors expand during the per-step attention computation.

---

## 8. Implementation Scope

**File:** `src/llm_engine/model/Mistral/attention.py`

**Key parameters:**

- `n_heads`: number of query heads (e.g., 32 for 7B, 64 for 70B)
- `n_kv_heads`: number of KV heads (e.g., 8 for 7B, 8 for 70B)
- `head_dim`: `d_model // n_heads`
- `n_rep`: `n_heads // n_kv_heads` (group size for repeat_kv)

**Four `nn.Linear` projections (all `bias=False`):**

- `W_q`: projects to `n_heads * head_dim` — full query heads
- `W_k`: projects to `n_kv_heads * head_dim` — fewer KV heads
- `W_v`: projects to `n_kv_heads * head_dim` — fewer KV heads
- `W_o`: projects `n_heads * head_dim` back to `d_model`

**`repeat_kv(x, n_rep)`:**

- Insert a new dimension: `(B, T, n_kv_heads, head_dim)` → `(B, T, n_kv_heads, 1, head_dim)`
- Expand along that dimension: → `(B, T, n_kv_heads, n_rep, head_dim)`
- Reshape to merge: → `(B, T, n_heads, head_dim)`
- If `n_rep == 1` (MHA), return input unchanged

**Forward flow:**

- Project Q with `W_q`, reshape to `(B, T, n_heads, head_dim)`
- Project K, V with `W_k`, `W_v`, reshape to `(B, T, n_kv_heads, head_dim)`
- Apply RoPE to Q and K (position encoding)
- Update KV cache if decoding
- Expand K, V via `repeat_kv` to match Q's head count
- Standard scaled dot-product attention: `softmax(Q K^T / sqrt(d_h)) @ V`
- Apply causal mask, then output projection with `W_o`

**Differences from GPT-2 `MultiHeadAttention`:**

- Separate `W_q`, `W_k`, `W_v` instead of fused `c_attn` (needed because K/V have fewer heads)
- `repeat_kv` expansion before attention computation
- RoPE applied to Q/K instead of absolute position embeddings
- All projections `bias=False`
- No dropout (Mistral doesn't use attention dropout)
