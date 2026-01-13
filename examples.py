"""
Example demonstrations for the GPT-OSS-20B Transformer Architecture.

This script provides simple examples of how to use the model for various tasks.
"""

import torch
from model import GPT, create_gpt_oss_20b, MultiHeadAttention, FeedForward, TransformerBlock


def example_1_create_and_inspect_model():
    """Example 1: Create a model and inspect its components."""
    print("=" * 70)
    print("Example 1: Creating and Inspecting the Model")
    print("=" * 70)
    
    # Create model
    model = create_gpt_oss_20b(vocab_size=50257, max_seq_len=512)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size (approx): {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Inspect architecture
    print(f"\nModel Architecture:")
    print(f"  Embedding dimension (d_model): {model.d_model}")
    print(f"  Number of transformer blocks: {len(model.blocks)}")
    print(f"  Maximum sequence length: {model.max_seq_len}")
    
    return model


def example_2_forward_pass():
    """Example 2: Perform a forward pass through the model."""
    print("\n" + "=" * 70)
    print("Example 2: Forward Pass Through the Model")
    print("=" * 70)
    
    # Create a smaller model for faster demonstration
    model = GPT(
        vocab_size=1000,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1
    )
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input (first sequence): {input_ids[0].tolist()[:10]}...")
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(input_ids)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"  (batch_size, seq_len, vocab_size) = ({batch_size}, {seq_len}, {1000})")
    
    # Get predicted next tokens
    predicted_tokens = torch.argmax(logits, dim=-1)
    print(f"\nPredicted next tokens (first sequence): {predicted_tokens[0].tolist()[:10]}...")
    
    return model


def example_3_attention_mechanism():
    """Example 3: Demonstrate the multi-head attention mechanism."""
    print("\n" + "=" * 70)
    print("Example 3: Multi-Head Attention Mechanism")
    print("=" * 70)
    
    # Create attention layer
    d_model = 512
    num_heads = 8
    attention = MultiHeadAttention(d_model, num_heads, dropout=0.1)
    attention.eval()
    
    # Create input
    batch_size = 1
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nAttention Configuration:")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Dimension per head: {d_model // num_heads}")
    
    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    
    # Forward pass
    with torch.no_grad():
        output = attention(x, mask)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Causal mask shape: {mask.shape}")
    print(f"\nCausal mask (prevents attending to future positions):")
    print(mask[0, 0].numpy().astype(int))
    
    return attention


def example_4_text_generation():
    """Example 4: Generate text autoregressively."""
    print("\n" + "=" * 70)
    print("Example 4: Autoregressive Text Generation")
    print("=" * 70)
    
    # Create a small model
    model = GPT(
        vocab_size=100,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        max_seq_len=50,
        dropout=0.0
    )
    model.eval()
    
    # Starting sequence
    start_tokens = torch.tensor([[1, 2, 3, 4, 5]])
    print(f"\nStarting tokens: {start_tokens[0].tolist()}")
    
    # Generate with different strategies
    print("\n1. Greedy Generation (temperature=0.1):")
    with torch.no_grad():
        generated_greedy = model.generate(start_tokens, max_new_tokens=10, temperature=0.1)
    print(f"   Generated: {generated_greedy[0].tolist()}")
    
    print("\n2. Sampling Generation (temperature=1.0):")
    with torch.no_grad():
        generated_sample = model.generate(start_tokens, max_new_tokens=10, temperature=1.0)
    print(f"   Generated: {generated_sample[0].tolist()}")
    
    print("\n3. Top-k Sampling (temperature=0.8, top_k=10):")
    with torch.no_grad():
        generated_topk = model.generate(start_tokens, max_new_tokens=10, temperature=0.8, top_k=10)
    print(f"   Generated: {generated_topk[0].tolist()}")
    
    print("\nNote: With a trained model on real data, these would be actual words!")
    
    return model


def example_5_training_step():
    """Example 5: Demonstrate a single training step."""
    print("\n" + "=" * 70)
    print("Example 5: Single Training Step")
    print("=" * 70)
    
    # Create model
    model = GPT(
        vocab_size=100,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_seq_len=32,
        dropout=0.1
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create dummy training data
    batch_size = 4
    seq_len = 20
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    target_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Learning rate: 1e-4")
    
    # Training step
    model.train()
    
    # Forward pass
    logits, loss = model(input_ids, target_ids)
    
    print(f"\nBefore training step:")
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimization step
    optimizer.step()
    
    # Forward pass after update
    with torch.no_grad():
        logits, new_loss = model(input_ids, target_ids)
    
    print(f"After training step:")
    print(f"  Loss: {new_loss.item():.4f}")
    print(f"  Loss change: {new_loss.item() - loss.item():.4f}")
    
    return model


def example_6_model_components():
    """Example 6: Inspect individual model components."""
    print("\n" + "=" * 70)
    print("Example 6: Understanding Model Components")
    print("=" * 70)
    
    # 1. Feed-Forward Network
    print("\n1. Feed-Forward Network:")
    ff = FeedForward(d_model=512, d_ff=2048, dropout=0.1)
    x = torch.randn(1, 10, 512)
    output = ff(x)
    print(f"   Input: {x.shape} -> Output: {output.shape}")
    print(f"   Expands to d_ff={2048}, then projects back to d_model={512}")
    
    # 2. Transformer Block
    print("\n2. Transformer Block:")
    block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048, dropout=0.1)
    x = torch.randn(1, 10, 512)
    mask = torch.tril(torch.ones(10, 10)).view(1, 1, 10, 10)
    output = block(x, mask)
    print(f"   Input: {x.shape} -> Output: {output.shape}")
    print(f"   Contains: Attention -> Add&Norm -> FFN -> Add&Norm")
    
    # 3. Embeddings
    print("\n3. Token and Position Embeddings:")
    vocab_size = 50257
    d_model = 768
    max_seq_len = 1024
    token_emb = torch.nn.Embedding(vocab_size, d_model)
    pos_emb = torch.nn.Embedding(max_seq_len, d_model)
    
    tokens = torch.randint(0, vocab_size, (1, 10))
    positions = torch.arange(0, 10).unsqueeze(0)
    
    tok_embeddings = token_emb(tokens)
    pos_embeddings = pos_emb(positions)
    combined = tok_embeddings + pos_embeddings
    
    print(f"   Token embeddings: {tok_embeddings.shape}")
    print(f"   Position embeddings: {pos_embeddings.shape}")
    print(f"   Combined: {combined.shape}")
    
    print("\n" + "=" * 70)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GPT-OSS-20B Transformer Architecture - Examples")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    example_1_create_and_inspect_model()
    example_2_forward_pass()
    example_3_attention_mechanism()
    example_4_text_generation()
    example_5_training_step()
    example_6_model_components()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Load real text data and tokenize it")
    print("  2. Train the model using train.py")
    print("  3. Generate text using the trained model")
    print("  4. Experiment with different hyperparameters")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
