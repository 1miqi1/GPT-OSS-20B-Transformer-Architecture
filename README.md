# GPT-OSS-20B Transformer Architecture

A comprehensive implementation of a decoder-only transformer architecture similar to GPT models, created for educational purposes to understand the inner workings of modern large language models.

## Overview

This repository implements the core components of a GPT-style transformer model, including:

- **Multi-Head Self-Attention**: Allows the model to attend to different parts of the input sequence
- **Position-wise Feed-Forward Networks**: Applies transformations to each position independently
- **Layer Normalization**: Stabilizes training by normalizing activations
- **Positional Encodings**: Provides sequence position information to the model
- **Causal Masking**: Ensures autoregressive generation (each token only attends to previous tokens)

## Architecture Details

### Transformer Block

Each transformer block consists of:

1. **Pre-Layer Normalization**: Applied before the attention layer
2. **Multi-Head Self-Attention**: 
   - Projects input into Query, Key, Value representations
   - Computes scaled dot-product attention across multiple heads
   - Uses causal masking for autoregressive modeling
3. **Residual Connection**: Adds input to attention output
4. **Pre-Layer Normalization**: Applied before the feed-forward layer
5. **Feed-Forward Network**: 
   - Two linear layers with GELU activation
   - Typically expands to 4x the model dimension
6. **Residual Connection**: Adds input to feed-forward output

### Model Components

```
Input Tokens
    ↓
Token Embedding + Position Embedding
    ↓
[Transformer Block] × N layers
    ↓
Layer Normalization
    ↓
Linear Projection to Vocabulary
    ↓
Output Logits
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- PyTorch 2.0+
- NumPy 1.24+

## Usage

### Basic Model Creation

```python
from model import create_gpt_oss_20b

# Create a GPT model
model = create_gpt_oss_20b(vocab_size=50257, max_seq_len=1024)

# Print model information
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Training

Run the training script:

```bash
python train.py
```

The training script demonstrates:
- Creating training and validation datasets
- Language modeling objective (predicting next token)
- Training loop with gradient clipping
- Evaluation with perplexity metric

### Custom Training

```python
from model import create_gpt_oss_20b
from train import train_model

# Create model
model = create_gpt_oss_20b()

# Prepare your tokenized data (list of token IDs)
train_tokens = [...]  # Your training data
val_tokens = [...]    # Your validation data

# Train
trained_model = train_model(
    model=model,
    train_data=train_tokens,
    val_data=val_tokens,
    seq_len=128,
    batch_size=32,
    num_epochs=10,
    learning_rate=3e-4
)
```

### Text Generation

```python
import torch
from model import create_gpt_oss_20b

# Load model
model = create_gpt_oss_20b()
model.eval()

# Starting sequence (tokenized)
start_tokens = torch.tensor([[1, 2, 3, 4, 5]])

# Generate new tokens
generated = model.generate(
    start_tokens,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40
)

print(f"Generated tokens: {generated[0].tolist()}")
```

## Model Configuration

The default configuration is a scaled-down version suitable for educational purposes:

```python
config = {
    'vocab_size': 50257,      # GPT-2 tokenizer vocabulary
    'd_model': 768,            # Model/embedding dimension
    'num_layers': 12,          # Number of transformer blocks
    'num_heads': 12,           # Number of attention heads
    'd_ff': 3072,              # Feed-forward dimension (4 × d_model)
    'max_seq_len': 1024,       # Maximum sequence length
    'dropout': 0.1             # Dropout probability
}
```

For a larger model closer to GPT-3 scale, you can adjust:
- `d_model`: 6144 or higher
- `num_layers`: 44 or more
- `num_heads`: 64 or more
- `d_ff`: 4 × d_model

**Note**: The actual GPT-3 20B model has approximately 20 billion parameters. This implementation can be scaled up by adjusting the configuration parameters.

## Key Concepts

### Language Modeling Objective

The model is trained to predict the next token in a sequence, given all previous tokens. This is formalized as:

```
P(x_t | x_1, x_2, ..., x_{t-1})
```

The loss function is cross-entropy between predicted and actual next tokens:

```
Loss = -Σ log P(x_t | x_1, ..., x_{t-1})
```

### Scaled Dot-Product Attention

The attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q (Query): What we're looking for
- K (Key): What each position offers
- V (Value): The actual content to retrieve
- d_k: Dimension of key vectors (used for scaling)

### Causal Masking

To ensure autoregressive generation, we apply a causal mask that prevents positions from attending to future positions. This is implemented as a lower triangular matrix:

```
Mask = [[1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]]
```

## File Structure

```
.
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── model.py           # GPT model implementation
└── train.py           # Training script and utilities
```

## Learning Resources

To better understand the implementation:

1. **Attention Mechanism**: Read "Attention is All You Need" (Vaswani et al., 2017)
2. **GPT Architecture**: Review the GPT-2 and GPT-3 papers by OpenAI
3. **Layer Normalization**: Understanding pre-norm vs post-norm architectures
4. **Positional Encoding**: Why position information is crucial for transformers

## Implementation Notes

### Design Decisions

1. **Pre-Norm Architecture**: Layer normalization is applied before attention and feed-forward layers, which improves training stability for deep models.

2. **GELU Activation**: We use GELU (Gaussian Error Linear Unit) instead of ReLU, as it's used in GPT models and provides smoother gradients.

3. **Weight Initialization**: Weights are initialized with a normal distribution (mean=0, std=0.02) following GPT practices.

4. **Gradient Clipping**: Applied during training to prevent exploding gradients, especially important for deep networks.

### Performance Considerations

- The model supports both CPU and GPU training
- Gradient clipping prevents instability during training
- Dropout is applied for regularization
- The implementation uses PyTorch's efficient built-in operations

## Extending the Model

You can extend this implementation by:

1. **Adding More Features**:
   - Byte-pair encoding (BPE) tokenization
   - Learning rate scheduling
   - Mixed precision training
   - Distributed training support

2. **Architecture Variations**:
   - Sparse attention patterns
   - Relative positional encodings
   - Different activation functions

3. **Training Improvements**:
   - Gradient accumulation for larger effective batch sizes
   - Curriculum learning
   - Data augmentation techniques

## License

This implementation is for educational purposes.

## Acknowledgments

This implementation is inspired by:
- The original Transformer architecture (Vaswani et al., 2017)
- OpenAI's GPT series of models
- Various open-source transformer implementations

## Contributing

This is an educational project. Feel free to use it for learning and experimentation!