"""
GPT-OSS-20B Transformer Architecture Implementation

This module implements a decoder-only transformer architecture similar to GPT models.
The implementation includes multi-head self-attention, feed-forward networks, and
layer normalization components that form the core of modern language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    This layer applies scaled dot-product attention across multiple heads,
    allowing the model to attend to information from different representation
    subspaces at different positions.
    
    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for queries, keys, values, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor for causal attention
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for causal/masked attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(attention_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This is a simple two-layer feed-forward network with GELU activation,
    applied to each position independently and identically.
    
    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward layer (typically 4 * d_model)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation as used in GPT models
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block.
    
    This block consists of:
    1. Masked multi-head self-attention with residual connection and layer norm
    2. Feed-forward network with residual connection and layer norm
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward layer
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask for causal attention
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-norm architecture: LayerNorm -> Attention -> Residual
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm architecture: LayerNorm -> FFN -> Residual
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x


class GPT(nn.Module):
    """
    GPT-OSS-20B: A decoder-only transformer language model.
    
    This model implements a stack of transformer decoder blocks for language modeling.
    It uses causal (autoregressive) masking to ensure each position can only attend
    to previous positions in the sequence.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model (embedding dimension)
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward layer
        max_seq_len: Maximum sequence length
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Optional target token indices for computing loss
            
        Returns:
            If targets is None: logits of shape (batch_size, seq_len, vocab_size)
            If targets is provided: tuple of (logits, loss)
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # Generate position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)  # (batch_size, seq_len, d_model)
        pos_emb = self.position_embedding(pos)  # (1, seq_len, d_model)
        x = self.dropout(tok_emb + pos_emb)
        
        # Create causal mask (lower triangular matrix)
        # This ensures each position can only attend to previous positions
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).view(1, 1, seq_len, seq_len)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (batch_size * seq_len, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
            
        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds max sequence length
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token to the sequence
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


def create_gpt_oss_20b(vocab_size=50257, max_seq_len=1024):
    """
    Create a GPT-OSS-20B model configuration.
    
    Note: The actual GPT-20B has ~20 billion parameters. This is a scaled-down
    version for educational purposes. You can adjust the parameters for different
    model sizes.
    
    Args:
        vocab_size: Size of the vocabulary (default: 50257 for GPT-2 tokenizer)
        max_seq_len: Maximum sequence length (default: 1024)
    
    Returns:
        GPT model instance
    """
    # Configuration similar to GPT-3 architecture patterns
    # For actual 20B model: d_model=6144, num_layers=44, num_heads=64
    # Scaled down for practical training:
    config = {
        'vocab_size': vocab_size,
        'd_model': 768,           # Model dimension
        'num_layers': 12,          # Number of transformer blocks
        'num_heads': 12,           # Number of attention heads
        'd_ff': 3072,              # Feed-forward dimension (4 * d_model)
        'max_seq_len': max_seq_len,
        'dropout': 0.1
    }
    
    model = GPT(**config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params:,} parameters")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating GPT-OSS-20B model...")
    model = create_gpt_oss_20b()
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, 50257, (batch_size, seq_len))
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    logits, _ = model(dummy_input)
    print(f"Output logits shape: {logits.shape}")
    
    # Test generation
    print("\nTesting text generation...")
    generated = model.generate(dummy_input[:1, :5], max_new_tokens=10)
    print(f"Generated sequence shape: {generated.shape}")
    
    print("\nModel successfully tested!")
