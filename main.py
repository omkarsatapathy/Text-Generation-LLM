# Add these imports at the top of main.py alongside existing imports
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os  # Make sure this is imported

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import tiktoken
import numpy as np
from pathlib import Path
import tiktoken
import logging
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from utils import GPTDecoder, TrainingLogger, TextDataset, create_dataloaders, train_epoch, create_optimizer, save_checkpoint, get_linear_schedule_with_warmup, evaluate
from config import Config
from torch.nn import LayerNorm
from typing import Tuple
from generate import generate_text_from_model
tokenizer = tiktoken.get_encoding("gpt2")


def optimize_config_for_memory(config: Config) -> Config:
    """Optimize configuration for memory efficiency"""
    # Reduce batch size
    config.training.batch_size = 4  # Smaller batch size
    
    # Reduce sequence length if it's too large
    config.model.max_seq_len = min(512, config.model.max_seq_len)
    config.data.max_length = min(512, config.data.max_length)
    
    # Reduce model size if needed
    if config.model.d_model > 512:
        config.model.d_model = 512
        config.model.ffn_hidden = 2048  # 4x d_model
        config.model.num_head = 8
    
    # Adjust number of layers if needed
    if config.model.num_layers > 8:
        config.model.num_layers = 8
    
    # Enable gradient checkpointing for memory efficiency
    config.model.use_gradient_checkpointing = True
    
    # Reduce number of workers to minimize memory usage
    config.training.num_workers = 2
    
    # Enable mixed precision training
    config.training.use_mixed_precision = True
    
    return config

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(
        "nccl",  # Use NCCL backend for distributed GPU training
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()
    
    
def main_worker(rank, world_size, args):
    """Main worker function for both distributed and non-distributed training"""
    is_distributed = rank is not None and world_size is not None
    
    if is_distributed:
        setup(rank, world_size)
    
    # Load and optimize configuration
    config = Config.load(args.config) if Path(args.config).exists() else Config()
    if args.data_path:
        config.data.data_path = Path(args.data_path)
    
    # Optimize config for memory
    config = optimize_config_for_memory(config)
    
    # Update device configuration
    if is_distributed:
        config.training.device = f'cuda:{rank}'
    device = torch.device(config.training.device)
    
    # Create model and move to device
    model = GPTDecoder(config).to(device)
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Wrap model in DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    should_log = not is_distributed or (is_distributed and rank == 0)
    training_logger = TrainingLogger(log_dir=output_dir / 'logs') if should_log else None
    
    # Create dataloaders with reduced batch size
    train_loader, val_loader = create_dataloaders(config, rank, world_size)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config)
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.training.use_mixed_precision else None
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.training.epochs):
        if should_log:
            logger.info(f"\nEpoch {epoch + 1}/{config.training.epochs}")
        
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        
        # Training loop with mixed precision
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, (x, y) in enumerate(progress):
            x, y = x.to(device), y.to(device)
            
            # Mixed precision training step
            with torch.amp.autocast('cuda',enabled=config.training.use_mixed_precision):
                loss = model(x, labels=y)
                loss = loss / config.training.gradient_accumulation_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            else:
                loss.backward()
                if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_value)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        
        if should_log:
            logger.info(f"Train loss: {avg_loss:.4f}")
            
            # Evaluate with mixed precision
            model.eval()
            val_loss = 0
            with torch.no_grad(), torch.amp.autocast('cuda',enabled=config.training.use_mixed_precision):
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    loss = model(x, labels=y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            model_to_save = model.module if is_distributed else model
            save_checkpoint(
                model_to_save, optimizer, scheduler, config,
                epoch, val_loss, output_dir, is_best
            )
            
            training_logger.plot_metrics()
    
    if is_distributed:
        cleanup()
        
        

def print_model_info(model: nn.Module, rank: int = 0):
    """Print model information for debugging"""
    if rank == 0:  # Only print from main process
        print("\nModel Information:")
        actual_model = model.module if isinstance(model, DDP) else model
        
        print(f"Model type: {type(model)}")
        if isinstance(model, DDP):
            print(f"Base model type: {type(actual_model)}")
        
        print("\nModel parameters:")
        total_params = sum(p.numel() for p in actual_model.parameters())
        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\nModel structure:")
        for name, module in actual_model.named_children():
            print(f"{name}: {type(module)}")
            
        print("\nDevice information:")
        print(f"Current device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")



def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='GPT Training Script')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                      help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--distributed', action='store_true',
                      help='Enable distributed training')
    args = parser.parse_args()
    
    if args.distributed:
        # Distributed training mode
        world_size = torch.cuda.device_count()
        print(f"Found {world_size} GPUs!")
        if world_size > 1:
            mp.spawn(
                main_worker,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
        else:
            print("Less than 2 GPUs available. Running in single GPU mode.")
            main_worker(0, 1, args)
    else:
        # Single GPU training mode
        main_worker(None, None, args)
    
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs!")
    
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
        
    # Load configuration
    config = Config.load(args.config) if Path(args.config).exists() else Config()
    if args.data_path:
        config.data.data_path = Path(args.data_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    training_logger = TrainingLogger(log_dir=output_dir / 'logs')

    # Set device
    device = torch.device(config.training.device)
    logger.info(f"Using device: {device}")

    # Create model and move to device
    model = GPTDecoder(config).to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.training.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.training.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch, training_logger
        )
        logger.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(
            model, val_loader, config, training_logger,
            step=epoch * len(train_loader)
        )
        logger.info(f"Validation loss: {val_loss:.4f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss



        # Generate sample text periodically
        if (epoch + 1) % 5 == 0:  # Every 5 epochs
            sample_text = generate_text_from_model(
                model,
                tokenizer = tokenizer,
                prompt="Once upon a time",
                # config=config
            )
            logger.info(f"\nSample generation:\n{sample_text}")
            print(f'Sample generated text : {sample_text}')
            save_checkpoint(
                model, optimizer, scheduler, config,
                epoch, val_loss, output_dir, is_best
            )

        # Plot metrics
        training_logger.plot_metrics()

    final_path = output_dir / 'final_model.pt'
    save_checkpoint(
        model, optimizer, scheduler, config,
        config.training.epochs - 1, val_loss,
        output_dir, is_best=False
    )
    logger.info("Training complete!")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
    
    
    
    
    