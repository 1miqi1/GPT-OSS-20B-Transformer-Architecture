"""
Training script for GPT-OSS-20B model.

This script demonstrates how to train the GPT model using a language modeling
objective on text data.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import create_gpt_oss_20b
import time


class TextDataset(Dataset):
    """
    Simple dataset for language modeling.
    
    This dataset takes a sequence of tokens and creates training examples
    by sliding a window across the sequence.
    
    Args:
        tokens: List or tensor of token indices
        seq_len: Length of each training sequence
    """
    
    def __init__(self, tokens, seq_len):
        # Convert to tensor once during initialization for efficiency
        if not isinstance(tokens, torch.Tensor):
            self.tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        """
        Returns a sequence and its target (shifted by one position).
        
        For language modeling, the target is the input shifted by one position.
        The model learns to predict the next token given the previous tokens.
        """
        # Use tensor slicing for efficiency - no conversion overhead
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return x, y


def create_sample_data(vocab_size=50257, num_tokens=10000):
    """
    Create sample random data for demonstration.
    
    In practice, you would load real text data, tokenize it, and use that instead.
    
    Args:
        vocab_size: Size of the vocabulary
        num_tokens: Number of tokens to generate
        
    Returns:
        Tensor of token indices
    """
    return torch.randint(0, vocab_size, (num_tokens,))


def train_epoch(model, dataloader, optimizer, device, clip_grad=1.0):
    """
    Train the model for one epoch.
    
    Args:
        model: GPT model to train
        dataloader: DataLoader providing training data
        optimizer: Optimizer for updating model parameters
        device: Device to train on (cuda/cpu)
        clip_grad: Gradient clipping value
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: GPT model to evaluate
        dataloader: DataLoader providing evaluation data
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        Average loss and perplexity
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def train_model(
    model,
    train_data,
    val_data=None,
    seq_len=128,
    batch_size=32,
    num_epochs=5,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Main training loop for the GPT model.
    
    Args:
        model: GPT model to train
        train_data: List of training token indices
        val_data: Optional list of validation token indices
        seq_len: Sequence length for training
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
    """
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_data, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if val_data is not None:
        val_dataset = TextDataset(val_data, seq_len)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer (AdamW is commonly used for transformer models)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training batches: {len(train_loader)}")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Perplexity: {torch.exp(torch.tensor(train_loss)):.2f}")
        
        # Evaluate on validation set if provided
        if val_loader is not None:
            val_loss, val_perplexity = evaluate(model, val_loader, device)
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Perplexity: {val_perplexity:.2f}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    
    return model


if __name__ == "__main__":
    print("="*60)
    print("GPT-OSS-20B Training Script")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    print("\nCreating model...")
    model = create_gpt_oss_20b(vocab_size=50257, max_seq_len=128)
    
    # Create sample data
    # In practice, replace this with real tokenized text data
    print("\nGenerating sample training data...")
    train_tokens = create_sample_data(vocab_size=50257, num_tokens=10000)
    val_tokens = create_sample_data(vocab_size=50257, num_tokens=2000)
    
    # Train the model
    print("\nStarting training...")
    trained_model = train_model(
        model=model,
        train_data=train_tokens,
        val_data=val_tokens,
        seq_len=128,
        batch_size=8,
        num_epochs=2,
        learning_rate=3e-4
    )
    
    # Save the trained model
    print("\nSaving model...")
    torch.save(trained_model.state_dict(), 'gpt_model.pt')
    print("Model saved to gpt_model.pt")
    
    # Test generation
    print("\n" + "="*60)
    print("Testing text generation...")
    print("="*60)
    
    trained_model.eval()
    device = next(trained_model.parameters()).device
    
    # Generate from a random starting sequence
    start_tokens = torch.randint(0, 50257, (1, 10)).to(device)
    print(f"\nStarting sequence (token IDs): {start_tokens[0].tolist()}")
    
    with torch.no_grad():
        generated = trained_model.generate(start_tokens, max_new_tokens=20, temperature=1.0)
    
    print(f"Generated sequence (token IDs): {generated[0].tolist()}")
    print("\nNote: In practice, you would use a tokenizer to convert these IDs back to text.")
