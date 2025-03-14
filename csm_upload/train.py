import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Model, ModelArgs, llama3_2_1B, llama3_2_100M
from generator import Generator
from dataclasses import dataclass
from typing import List, Optional

# Add the new model definition
def llama3_2_3B() -> nn.Module:
    """Llama 3.2 3B model configuration."""
    from torchtune.models import llama3_2
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=8192,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

# Update FLAVORS dictionary
FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
    "llama-3B": llama3_2_3B,
}

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    backbone_flavor: str = "llama-3B"  # Use 3B model as default
    decoder_flavor: str = "llama-100M"
    text_vocab_size: int = 128256
    audio_vocab_size: int = 2051
    audio_num_codebooks: int = 32
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 5
    save_every: int = 1000
    eval_every: int = 500
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True
    load_checkpoint: Optional[str] = None
    compute_amortization: bool = True  # Enable compute amortization scheme
    amortization_factor: int = 16  # Train decoder on 1/16 of frames

def setup_model(config: TrainingConfig):
    """Initialize the model with the specified configuration."""
    model_args = ModelArgs(
        backbone_flavor=config.backbone_flavor,
        decoder_flavor=config.decoder_flavor,
        text_vocab_size=config.text_vocab_size,
        audio_vocab_size=config.audio_vocab_size,
        audio_num_codebooks=config.audio_num_codebooks,
    )
    
    model = Model(model_args)
    
    # Initialize with pre-trained weights if available
    if config.backbone_flavor == "llama-3B" and os.path.exists("Llama-3.2-3B-Instruct"):
        print("Loading pre-trained Llama-3.2-3B-Instruct weights...")
        # This is a placeholder - actual implementation would depend on how you want to load weights
        # You'll need to map the pre-trained weights to your model architecture
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train CSM model")
    parser.add_argument("--backbone", type=str, default="llama-3B", choices=FLAVORS.keys(), 
                        help="Backbone model flavor")
    parser.add_argument("--decoder", type=str, default="llama-100M", choices=FLAVORS.keys(),
                        help="Decoder model flavor")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--no_amortization", action="store_true", help="Disable compute amortization")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        backbone_flavor=args.backbone,
        decoder_flavor=args.decoder,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        load_checkpoint=args.checkpoint,
        fp16=args.fp16,
        compute_amortization=not args.no_amortization,
    )
    
    print(f"Setting up model with backbone: {config.backbone_flavor}, decoder: {config.decoder_flavor}")
    model = setup_model(config)
    
    # Move model to device and enable mixed precision if requested
    model = model.to(config.device)
    if config.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("Using mixed precision training")
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print("Model setup complete. Ready for training.")
    print(f"Using device: {config.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Training would be implemented here
    # This is a placeholder for the actual training loop
    print("Training not implemented yet. This script is a starting point.")

if __name__ == "__main__":
    main()
