# Transformer — How it works & Hyperparameter Guide

---

## How a Transformer works (from scratch)

### The core idea: attention

The key insight of the Transformer (Vaswani et al., 2017) is **self-attention**: instead of processing a sequence step by step (like an LSTM), every position in the sequence looks at every other position simultaneously and decides how much to "pay attention" to each one.

For gesture recognition, each timestep of the gesture trajectory can attend to every other timestep. This lets the model learn, for example, that the end of a gesture depends on how it started.

### Step-by-step walkthrough (this codebase)

```
Raw gesture trajectory
      ↓
1. Resample to fixed length (target_length timesteps)
      ↓
2. Normalize (zero mean, unit variance per feature)
      ↓
3. Input projection  →  Dense(d_model)
   Maps each timestep from raw feature space into the model's internal space.
      ↓
4. Prepend [CLS] token  (learnable vector, BERT-style)
   A special "summary" position prepended to the sequence.
   After encoding, the model reads only this position for classification.
      ↓
5. Add positional encoding
   Sinusoidal signals (tensor2tensor style) injected into the sequence
   so the model knows the ORDER of timesteps (attention alone is order-agnostic).
      ↓
6. Encoder block(s) — repeated num_layers times:

   ┌──────────────────────────────────────────────────────────┐
   │  LayerNorm(x)                  ← stabilizes inputs       │
   │       ↓                                                  │
   │  Multi-Head Self-Attention     ← each position attends   │
   │       ↓                          to all others           │
   │  Dropout + Residual            ← x = x + dropout(attn)  │
   │                                                          │
   │  LayerNorm(x)                                            │
   │       ↓                                                  │
   │  FFN: Dense(4×d_model, GELU) → Dropout → Dense(d_model) │
   │       ↓                                                  │
   │  Dropout + Residual            ← x = x + dropout(ffn)   │
   └──────────────────────────────────────────────────────────┘
      ↓
7. Final LayerNorm
      ↓
8. Read out the [CLS] token (position 0)
      ↓
9. Dropout → Dense(n_classes, softmax)
      ↓
   Predicted gesture class
```

### What is Multi-Head Attention?

Attention computes, for each position, a weighted sum of all other positions' values.
The weights are computed as:

```
Attention(Q, K, V) = softmax(Q × Kᵀ / sqrt(key_dim)) × V
```

- **Q (Query)**: "what am I looking for?"
- **K (Key)**: "what does each position offer?"
- **V (Value)**: "what information does each position carry?"

**Multi-head** means this is done `n_heads` times in parallel, each head learning a different type of relationship. The results are concatenated and projected back to `d_model`.

### What is the FFN?

After attention, each position is processed independently through a small 2-layer network:
- Expand: `d_model → 4×d_model` with GELU activation
- Compress: `4×d_model → d_model`

This is where the model does most of its "thinking" about individual features.

### What are residual connections?

At each sub-layer (attention and FFN), the **input is added back to the output**:
```
x = x + sublayer(x)
```
This lets gradients flow directly back through the network during training, making deep models trainable. Without them, transformers don't train.

### What is LayerNorm?

LayerNorm normalizes the activations at each position to have zero mean and unit variance. Applied **before** each sub-layer here (pre-norm, tensor2tensor default), it stabilizes training and allows higher learning rates.

---

## Hyperparameter Reference

### Architecture hyperparameters

#### `d_model` — tested: `[32, 64]`
The **width** of the model. Every timestep is represented as a vector of `d_model` numbers throughout the entire network.

| Value | Meaning |
|---|---|
| `32` | Small, fast, fewer parameters. Safer for small datasets. |
| `64` | Medium. Twice as many parameters. More expressive. |
| `128`+ | Large. Risk of overfitting on small datasets. |

The number of attention heads (`n_heads`) is auto-computed as the largest power of 2 ≤ 8 that divides `d_model` evenly. `key_dim = d_model / n_heads`.

---

#### `num_layers` — tested: `[1, 2]`
The **depth** — how many encoder blocks are stacked.

| Value | Meaning |
|---|---|
| `1` | One pass of attention + FFN. Simple, fast, good baseline. |
| `2` | Two stacked blocks. Can capture more abstract patterns. |
| `4`+ | Very deep for small data — typically overfits. |

---

#### `ffn_filter_size` — fixed at `4 × d_model`
The intermediate size of the feed-forward network inside each encoder block. `d_model=64` → FFN size = 256. This is the original Vaswani 2017 ratio and almost never changed.

---

#### `target_length` — tested: `[32, 64]`
Gestures have variable durations. This resamples every gesture to a fixed number of timesteps using linear interpolation.

| Value | Meaning |
|---|---|
| `32` | Coarser resolution. Faster, less memory. |
| `64` | Finer resolution. More temporal detail. |
| `128` | High detail. May not help if gestures are short or noisy. |

---

### Regularization hyperparameters

#### `dropout_rate` — fixed at `0.3`
During training, randomly sets a fraction of neuron outputs to zero. Prevents the model from memorizing the training data.

Applied in 4 places:
- On attention weights (inside MultiHeadAttention)
- Between the two FFN layers
- On residual connections (after attention and FFN)
- Before the final classifier

| Value | Meaning |
|---|---|
| `0.0` | No dropout. Fast convergence, likely overfits. |
| `0.1` | Light. Good for large datasets. |
| `0.3` | Moderate. Good default for small datasets (current). |
| `0.5` | Heavy. Use only if model badly overfits. |

---

#### L2 regularization — fixed at `1e-4`
Penalizes large weights in all Dense layers. Pushes the model toward simpler solutions.
`1e-4 = 0.0001`. Standard value, rarely needs tuning.

---

### Training hyperparameters

#### `epochs` — fixed at `40`
Maximum number of complete passes over the training data. In practice, EarlyStopping stops training before 40 if validation loss stagnates.

---

#### `batch_size` — fixed at `32`
Number of samples processed before each weight update.

| Value | Meaning |
|---|---|
| `8` | Very frequent, noisy updates. Slower but sometimes better generalization. |
| `16` | Small batch. Slight regularization effect from noise. |
| `32` | Standard balance between speed and noise (current). |
| `64` | Faster per epoch, smoother gradients, may overfit more. |

---

#### `validation_split` — fixed at `0.10`
10% of the training data is held out (per fold) to monitor validation loss. Used only by EarlyStopping — it never affects the test set.

---

#### `warmup_steps` — fixed at `200`
Controls the learning rate schedule (Vaswani 2017 §5.3):

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

- Steps 1 → 200: learning rate **increases linearly** (warmup — model is unstable early)
- Steps 200+: learning rate **decreases** as `1/sqrt(step)`

| Value | Effect |
|---|---|
| `100` | Short warmup. Reaches peak LR faster. Riskier. |
| `200` | Default. Good for small datasets (current). |
| `400` | Longer warmup. More conservative startup. |

---

#### EarlyStopping `patience=5`
If validation loss does not improve for 5 consecutive epochs, training stops and the best weights are restored. Prevents wasted compute and overfitting.

---

#### Adam optimizer — `beta_1=0.9, beta_2=0.98, epsilon=1e-9`
Values taken directly from the original Transformer paper. Almost never changed.

| Parameter | Meaning |
|---|---|
| `beta_1=0.9` | Momentum — smooths the gradient over time. |
| `beta_2=0.98` | Smooths the squared gradient (controls adaptive step size). |
| `epsilon=1e-9` | Prevents division by zero. |

---

## Is this state of the art?

**For gesture recognition on small datasets: no, but it is a solid and well-implemented baseline.**

**What's good:**
- Follows the exact tensor2tensor Vaswani 2017 spec (pre-normalization, correct LR schedule, correct positional encoding)
- BERT-style [CLS] token for classification — a proven technique
- Proper regularization stack (dropout + L2 + early stopping)

**What SOTA models add:**
- **Data augmentation**: time warping, jitter, random scaling. Transformers are data-hungry; this dataset is small.
- **Larger pre-trained backbones**: TimesFM, Moirai — overkill here but used in production systems.
- **Temporal Convolutional Networks (TCN)**: often outperform transformers on small gesture datasets because their local-temporal inductive bias matches gesture structure.
- **Learned positional embeddings**: sometimes outperform sinusoidal on short sequences.

---

## What to tune and why

| Priority | Hyperparameter | Suggested values | Why |
|---|---|---|---|
| **High** | `dropout_rate` | `[0.1, 0.2, 0.3, 0.4, 0.5]` | Biggest impact on small datasets. If val loss >> train loss → increase. |
| **High** | `d_model` | `[32, 64, 128]` | Controls model capacity. If underfitting → go bigger. |
| **High** | `target_length` | `[32, 64, 128]` | Affects temporal resolution. |
| **Medium** | `num_layers` | `[1, 2, 3]` | More layers = more abstraction, more overfitting risk. |
| **Medium** | `batch_size` | `[16, 32, 64]` | Affects gradient noise and training speed. |
| **Medium** | `warmup_steps` | `[100, 200, 400]` | If loss spikes early → increase warmup. |
| **Low** | `patience` | `[5, 10, 15]` | If training stops too early → increase. |
| **Low** | L2 `1e-4` | `[1e-5, 1e-4, 1e-3]` | Only tune if dropout alone isn't enough. |

**Start with `dropout_rate` and `d_model` — they have the most impact on small gesture datasets.**
