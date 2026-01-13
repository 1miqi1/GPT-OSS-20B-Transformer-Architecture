# GPT-OSS-20B Architecture Details

This document provides an in-depth explanation of the GPT-OSS-20B transformer architecture implementation.

## Table of Contents
1. [Overall Architecture](#overall-architecture)
2. [Multi-Head Attention](#multi-head-attention)
3. [Feed-Forward Networks](#feed-forward-networks)
4. [Transformer Block](#transformer-block)
5. [Embeddings](#embeddings)
6. [Training Objective](#training-objective)
7. [Mathematical Formulations](#mathematical-formulations)

## Overall Architecture

The GPT-OSS-20B model is a decoder-only transformer architecture. Here's the high-level flow:

```
Input Token IDs
       ↓
Token Embeddings (vocab_size → d_model)
       +
Position Embeddings (max_seq_len → d_model)
       ↓
    Dropout
       ↓
[Transformer Block 1]
       ↓
[Transformer Block 2]
       ↓
       ...
       ↓
[Transformer Block N]
       ↓
Layer Normalization
       ↓
Linear Projection (d_model → vocab_size)
       ↓
Output Logits
```

### Key Components

1. **Token Embedding Layer**: Maps discrete tokens to continuous vectors
2. **Positional Embedding**: Adds position information to tokens
3. **Transformer Blocks**: Stack of N identical blocks (12 in default config)
4. **Output Layer**: Projects back to vocabulary space

## Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces.

### Architecture

```
Input: x (batch_size, seq_len, d_model)
       ↓
┌──────┴──────┬──────┬──────┐
│      │      │      │      │
Wq     Wk     Wv     Wo     (Linear projections)
│      │      │      │
Q      K      V      
│      │      │
└──────┴──────┘
       ↓
Split into num_heads
       ↓
(batch_size, num_heads, seq_len, d_k)
       ↓
Scaled Dot-Product Attention
   QK^T / √d_k
       ↓
   Softmax
       ↓
   Attention(Q,K,V)
       ↓
Concatenate heads
       ↓
Output projection (Wo)
       ↓
Output: (batch_size, seq_len, d_model)
```

### Causal Masking

For autoregressive language modeling, we use a causal mask:

```
Position:  0  1  2  3  4
    0:    [1  0  0  0  0]   ← Can only see position 0
    1:    [1  1  0  0  0]   ← Can see positions 0-1
    2:    [1  1  1  0  0]   ← Can see positions 0-2
    3:    [1  1  1  1  0]   ← Can see positions 0-3
    4:    [1  1  1  1  1]   ← Can see positions 0-4
```

This prevents each position from attending to future positions.

### Attention Computation

For each head, the attention is computed as:

```python
scores = Q @ K.T / sqrt(d_k)           # (seq_len, seq_len)
scores = scores.masked_fill(mask == 0, -inf)  # Apply causal mask
attention_weights = softmax(scores)    # (seq_len, seq_len)
output = attention_weights @ V         # (seq_len, d_k)
```

## Feed-Forward Networks

The position-wise feed-forward network consists of two linear transformations with a GELU activation in between.

### Architecture

```
Input: x (batch_size, seq_len, d_model)
       ↓
Linear 1 (d_model → d_ff)
       ↓
GELU activation
       ↓
   Dropout
       ↓
Linear 2 (d_ff → d_model)
       ↓
Output: (batch_size, seq_len, d_model)
```

### Properties

- **Expansion**: Typically d_ff = 4 × d_model
- **Activation**: GELU (Gaussian Error Linear Unit) - smoother than ReLU
- **Independence**: Applied to each position independently

## Transformer Block

Each transformer block implements a pre-norm architecture with residual connections.

### Architecture

```
Input: x
       ↓
Layer Norm (ln1)
       ↓
Multi-Head Attention (with causal mask)
       ↓
   Dropout
       ↓
Residual: Add x
       ↓ (x')
Layer Norm (ln2)
       ↓
Feed-Forward Network
       ↓
   Dropout
       ↓
Residual: Add x'
       ↓
Output
```

### Pre-Norm vs Post-Norm

This implementation uses **pre-norm** (layer norm before attention/FFN):
- Better training stability for deep models
- Easier gradient flow
- Used in modern large language models (GPT-3, etc.)

## Embeddings

### Token Embeddings

Maps each token ID to a learned vector:

```
Token ID: 42
    ↓
Embedding Matrix[42] (lookup)
    ↓
Vector: [0.123, -0.456, 0.789, ...]  (d_model dimensions)
```

### Positional Embeddings

Adds learnable position information:

```
Position: 0, 1, 2, ..., seq_len-1
    ↓
Position Embedding Matrix[pos] (lookup)
    ↓
Vector: [0.321, 0.654, -0.987, ...]  (d_model dimensions)
```

The final input to the transformer is:
```
input_embedding = token_embedding + position_embedding
```

## Training Objective

### Language Modeling Loss

The model is trained to predict the next token given previous tokens.

```
Input:  [w1, w2, w3, w4, w5]
Target: [w2, w3, w4, w5, w6]
```

### Cross-Entropy Loss

For each position, we compute:

```
Loss = -Σ log P(w_target | w_1, ..., w_{t-1})
```

Where P is computed as:
```
logits = model(input_tokens)
probabilities = softmax(logits)
loss = cross_entropy(probabilities, target_tokens)
```

## Mathematical Formulations

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q: Query matrix (seq_len × d_k)
- K: Key matrix (seq_len × d_k)
- V: Value matrix (seq_len × d_k)
- d_k: Dimension of keys/queries

### Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Parameters:
- W_i^Q ∈ ℝ^(d_model × d_k)
- W_i^K ∈ ℝ^(d_model × d_k)
- W_i^V ∈ ℝ^(d_model × d_k)
- W_O ∈ ℝ^(h·d_k × d_model)

### Feed-Forward Network

```
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
```

Where:
- W_1 ∈ ℝ^(d_model × d_ff)
- W_2 ∈ ℝ^(d_ff × d_model)

### GELU Activation

```
GELU(x) = x · Φ(x)
```

Where Φ(x) is the cumulative distribution function of the standard Gaussian distribution.

### Layer Normalization

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

Where:
- μ: Mean across features
- σ²: Variance across features
- γ, β: Learned affine parameters
- ε: Small constant for numerical stability

### Complete Transformer Block

```
x' = x + Dropout(MultiHeadAttention(LayerNorm(x)))
output = x' + Dropout(FFN(LayerNorm(x')))
```

## Model Configurations

### Default Configuration (Educational)

```python
{
    'vocab_size': 50257,      # GPT-2 tokenizer vocabulary
    'd_model': 768,            # Embedding dimension
    'num_layers': 12,          # Number of transformer blocks
    'num_heads': 12,           # Attention heads
    'd_ff': 3072,              # FFN hidden size (4 × d_model)
    'max_seq_len': 1024,       # Maximum sequence length
    'dropout': 0.1             # Dropout rate
}
```

Parameters: ~163M (similar to GPT-2 medium)

### Scaling to 20B Parameters

To approximate a 20B parameter model:

```python
{
    'vocab_size': 50257,
    'd_model': 6144,           # Larger embedding
    'num_layers': 44,          # More layers
    'num_heads': 64,           # More attention heads
    'd_ff': 24576,             # 4 × d_model
    'max_seq_len': 2048,       # Longer context
    'dropout': 0.1
}
```

This would give approximately 20 billion parameters.

## Computational Complexity

### Attention Complexity

- Time: O(n² · d)
  - n: sequence length
  - d: model dimension
  
- Space: O(n²)
  - Stores attention matrix

### Feed-Forward Complexity

- Time: O(n · d · d_ff)
- Space: O(n · d_ff)

### Total per Layer

- Time: O(n² · d + n · d · d_ff)
- For long sequences: attention is the bottleneck (n²)
- For short sequences: FFN dominates (d_ff is typically 4d)

## Implementation Details

### Weight Initialization

```python
# Linear layers and embeddings
nn.init.normal_(weight, mean=0.0, std=0.02)
nn.init.zeros_(bias)  # if bias exists
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients during training.

### Optimizer Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

AdamW is the standard optimizer for transformer models.

## References

1. **Attention Is All You Need** - Vaswani et al., 2017
   - Original transformer architecture

2. **Improving Language Understanding by Generative Pre-Training** - Radford et al., 2018
   - GPT-1: First decoder-only transformer for language

3. **Language Models are Unsupervised Multitask Learners** - Radford et al., 2019
   - GPT-2: Scaling up decoder-only transformers

4. **Language Models are Few-Shot Learners** - Brown et al., 2020
   - GPT-3: Demonstrating the power of scale

5. **On Layer Normalization in the Transformer Architecture** - Xiong et al., 2020
   - Pre-norm vs post-norm analysis

## Key Insights

1. **Autoregressive Modeling**: Each token is predicted based only on previous tokens, making the model suitable for generation tasks.

2. **Parallel Training**: Despite autoregressive generation, training can be parallelized by computing all positions simultaneously with causal masking.

3. **Scaling Laws**: Model performance improves predictably with:
   - More parameters (larger d_model, more layers)
   - More training data
   - More compute

4. **Pre-norm Architecture**: Critical for training stability in very deep models (>12 layers).

5. **Positional Information**: Required because attention is permutation-invariant; the model needs explicit position signals.

## Common Questions

**Q: Why decoder-only instead of encoder-decoder?**
A: For language modeling, we only need to generate text autoregressively. Encoder-decoder is better for seq2seq tasks like translation.

**Q: Why GELU instead of ReLU?**
A: GELU provides smoother gradients and works better for language models in practice.

**Q: Why is the feed-forward network 4× larger?**
A: Empirically found to work well; provides capacity for the model to transform representations.

**Q: How does the model know word positions?**
A: Through learned positional embeddings added to token embeddings.

**Q: Can I use relative positions instead?**
A: Yes! Relative positional encodings (like in Transformer-XL) are an alternative approach.
