# SwiGLU (Swish-Gated Linear Unit)

## 1. The Problem: Why Standard FFN is Suboptimal

**GPT-2's FFN — Two linear layers with GELU activation:**

```python
h = GELU(x @ W_up) @ W_down
```

A standard FFN projects from `d_model` → `4*d_model` (expand), applies an activation, then projects back `4*d_model` → `d_model` (compress). All information flows through a single transform — there's no mechanism for the network to **selectively control** what information passes through.

**Problems with this approach:**

- **No gating mechanism** — every dimension is transformed uniformly by the activation. The network cannot suppress irrelevant features before they mix in the down projection.
- **GELU saturation** — GELU approaches zero for large negative inputs, but has no learned control over *which* dimensions get suppressed.
- **Parameter efficiency** — empirically, gated variants achieve better loss for the same compute budget (shown in the PaLM and LLaMA papers).

---

## 2. SwiGLU's Key Insight

Instead of one up-projection followed by an activation, SwiGLU splits the work into **two parallel projections** — one produces the content, the other produces a gate that controls how much of that content passes through.

$$\text{SwiGLU}(x) = (\text{Swish}(x W_{\text{gate}})) \odot (x W_{\text{up}})$$

The gate projection decides **what to keep**. The up projection decides **what to propose**. Element-wise multiplication combines them — only features that the gate "approves" survive to the down projection.

**Why this matters:**

- The model learns to selectively filter information at each FFN layer
- Two separate projections give the network more expressive power per parameter
- Swish (SiLU) is a smooth, non-monotonic activation — unlike ReLU, it can assign small negative values, providing richer gradients

---

## 3. How SwiGLU Works — Step by Step

### Step 1: Three linear projections (no bias)

Mistral's SwiGLU uses three weight matrices, all without bias:

- $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ — gate projection
- $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ — up projection
- $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ — down projection

where $d_{\text{ff}}$ is the intermediate/hidden FFN dimension. In Mistral 7B: $d_{\text{model}} = 4096$, $d_{\text{ff}} = 14336$.

### Step 2: Compute gate and up in parallel

For input $x$ of shape $(B, T, d_{\text{model}})$:

$$\text{gate} = \text{Swish}(x \cdot W_{\text{gate}}) \quad \text{shape: } (B, T, d_{\text{ff}})$$
$$\text{up} = x \cdot W_{\text{up}} \quad \text{shape: } (B, T, d_{\text{ff}})$$

The Swish (SiLU) activation is:

$$\text{Swish}(z) = z \cdot \sigma(z) = z \cdot \frac{1}{1 + e^{-z}}$$

### Step 3: Element-wise gating

$$\text{hidden} = \text{gate} \odot \text{up} \quad \text{shape: } (B, T, d_{\text{ff}})$$

The gate values (after Swish) modulate each dimension of the up projection. Dimensions where the gate is near zero get suppressed; dimensions where the gate is large pass through.

### Step 4: Down projection

$$\text{output} = \text{hidden} \cdot W_{\text{down}} \quad \text{shape: } (B, T, d_{\text{model}})$$

Projects back to model dimension.

---

## 4. Why Three Projections Instead of Two?

### GPT-2 FFN (standard)

```bash
x → [W_up: d → 4d] → GELU → [W_down: 4d → d] → output
```

**2 weight matrices**, parameter count: $d \times 4d + 4d \times d = 8d^2$

### Mistral SwiGLU

```bash
x → [W_gate: d → d_ff] → Swish ─┐
x → [W_up:   d → d_ff] ─────────┤ element-wise multiply
                                  ↓
                          [W_down: d_ff → d] → output
```

**3 weight matrices**, parameter count: $3 \times d \times d_{\text{ff}}$

To keep the total parameter count roughly **equal** to the standard FFN ($8d^2$), Mistral uses $d_{\text{ff}} = \frac{7}{2} \times d$ (14336 for d=4096), keeping roughly the same parameter budget as a standard 4d FFN while accounting for three projections.

For Mistral 7B: $d = 4096$, $d_{\text{ff}} = 14336 = 3.5 \times 4096$.

This keeps parameter parity: $3 \times 4096 \times 14336 \approx 1.31 \times 8 \times 4096^2$.

---

## 5. SwiGLU vs Other FFN Variants

| Method       | Activation       | Gating  | Projections            | Used In                          |
|--------------|------------------|---------|------------------------|----------------------------------|
| Standard FFN | GELU / ReLU      | No      | 2 (up, down)           | GPT-2, BERT, Falcon           |
| GLU          | Sigmoid          | Yes     | 3 (gate, up, down)     | Original GLU paper               |
| GeGLU        | GELU             | Yes     | 3 (gate, up, down)     | Some T5 variants                 |
| ReGLU        | ReLU             | Yes     | 3 (gate, up, down)     | Explored in Noam Shazeer's paper |
| **SwiGLU**   | **Swish (SiLU)** | **Yes** | **3 (gate, up, down)** | **LLaMA, PaLM, Mistral**         |

**Key differences from GPT-2:**

- **Gated architecture** — selective information filtering vs uniform activation
- **Three projections, no bias** — gate_proj, up_proj, down_proj; all `bias=False`
- **Swish activation** — smooth, non-monotonic (`z * sigmoid(z)`) vs GELU
- ****Adjusted d_ff** — 3.5 × d (14336 for Mistral 7B) to balance parameter budget with standard 4d

---

## 6. Interview-Ready Points

### Q: "Why does Mistral use SwiGLU instead of a standard GELU FFN?"

> SwiGLU introduces a **gating mechanism** — two parallel projections where one (after Swish activation) controls which dimensions of the other survive. This gives the network element-wise control over information flow. Empirically, gated variants like SwiGLU achieve better training loss per compute unit than standard FFNs. The tradeoff is three weight matrices instead of two, so the intermediate dimension is reduced to ⅔ × 4d to keep total parameters constant.

### Q: "What is the Swish activation and why use it over ReLU or GELU?"

> Swish is $z \cdot \sigma(z)$, also called SiLU. Unlike ReLU, it's smooth and non-monotonic — it can produce small negative outputs for slightly negative inputs, which provides richer gradient signal. Unlike GELU, it's simpler to compute (just sigmoid, no erf). Ablation studies in the Noam Shazeer paper showed SwiGLU outperformed GeGLU, ReGLU, and standard GELU on language modeling benchmarks.

### Q: "Why is d_ff = 14336 in Mistral 7B instead of 4 × 4096 = 16384?"

> SwiGLU has 3 weight matrices instead of 2. To keep the **total parameter count** equal to a standard FFN (which uses d_ff = 4d with 2 matrices, giving 8d² params), Mistral sets d_ff = 14336 (3.5 × d). This is larger than LLaMA 7B's 11008 but still keeps total params manageable with parameter-efficient SwiGLU gating.

### Q: "Why no bias in Mistral's linear layers?"

> Removing bias is a deliberate design choice. Biases add minimal parameters but can hurt training stability at scale — they create offset terms that interact poorly with weight decay and normalization. LLaMA, PaLM, and most modern LLMs remove all biases from attention and FFN projections. This also simplifies parallelism (no bias reduction across tensor-parallel ranks).

---

## 7. Implementation Scope

**File:** `src/llm_engine/model/Mistral/feedforward.py`

**Architecture:**

```bash
x ─→ gate_proj(x) ─→ Swish ──┐
  └→ up_proj(x) ──────────────┤  element-wise multiply
                               ↓
                         down_proj(hidden) ─→ output
```

**Three `nn.Linear` layers (all `bias=False`):**

- `gate_proj`: `nn.Linear(d_model, d_ff, bias=False)`
- `up_proj`: `nn.Linear(d_model, d_ff, bias=False)`
- `down_proj`: `nn.Linear(d_ff, d_model, bias=False)`

**Forward:**

```python
def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**Shape flow:** `(B, T, d_model)` → `(B, T, d_ff)` → `(B, T, d_model)`

**Called in Mistral block:** After attention + residual, before the second residual connection:

```python
x = x + feedforward(rmsnorm(x))
```
