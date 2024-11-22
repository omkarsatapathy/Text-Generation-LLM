import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math
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
from config import Config
from torch.nn import LayerNorm
from typing import Tuple
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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


@dataclass
class AttentionOutput:
    """Container for attention output"""
    attention_output: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


class LayerNorm(nn.Module):
    """Layer normalization with optional bias"""

    def __init__(self, d_model: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)
        if self.beta is None:
            return self.gamma * normalized
        return self.gamma * normalized + self.beta


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional flash attention support"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.model.num_head
        self.d_model = config.model.d_model
        self.head_dim = self.d_model // self.num_heads
        self.scaling = self.head_dim ** -0.5

        # Single linear layer for all projections
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(config.model.drop_prob)
        self.resid_dropout = nn.Dropout(config.model.drop_prob)

        # Flash attention flag
        self.use_flash = config.model.use_flash_attention

        # Initialize weights
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            need_weights: bool = False
    ) -> AttentionOutput:
        batch_size, seq_len, _ = x.shape

        # Calculate Q, K, V projections simultaneously
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0's native flash attention
            attention_mask = torch.ones(
                (batch_size, self.num_heads, seq_len, seq_len),
                device=x.device,
                dtype=torch.bool
            ).tril_()  # Create causal mask
            attention_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Manual implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask, float('-inf'))

            # Apply attention mask if provided
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))

            # Attention weights
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.attn_dropout(attention_weights)

            # Compute attention output
            attention_output = torch.matmul(attention_weights, v)

        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attention_output = self.out_proj(attention_output)
        attention_output = self.resid_dropout(attention_output)

        if need_weights:
            return AttentionOutput(attention_output, attention_weights if not self.use_flash else None)
        return AttentionOutput(attention_output)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.model.d_model, config.model.ffn_hidden)
        self.fc2 = nn.Linear(config.model.ffn_hidden, config.model.d_model)
        self.dropout = nn.Dropout(config.model.drop_prob)
        self.activation = nn.GELU()

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)  # Additional dropout for decoder
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer with causal attention"""

    def __init__(self, config):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.feed_forward = FeedForward(config)

        # Layer normalization layers (Pre-LN architecture)
        self.norm1 = LayerNorm(config.model.d_model)
        self.norm2 = LayerNorm(config.model.d_model)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN architecture
        # Layer 1: Causal self-attention
        residual = x
        x = self.norm1(x)
        attention_output = self.attention(x, attention_mask)
        x = residual + attention_output.attention_output

        # Layer 2: Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.feed_forward(x)

        return x


class GPTDecoder(nn.Module):
    """GPT-style decoder-only transformer"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.model.vocab_size, config.model.d_model)

        # Position embedding
        self.position_embedding = nn.Embedding(config.model.max_seq_len, config.model.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.model.drop_prob)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.model.num_layers)
        ])

        # Final layer normalization
        self.norm = LayerNorm(config.model.d_model)

        # Output projection (weight tying happens in forward)
        self.lm_head = lambda x: F.linear(x, self.token_embedding.weight)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution"""

        def _normal_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(_normal_init)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get sequence length and create position ids
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embed tokens and positions
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(position_ids)
        x = self.dropout(x)

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Apply final layer normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Calculate loss if labels are provided
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss

        return logits

    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text using various decoding strategies"""
        self.eval()
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]

        with torch.no_grad():
            for _ in range(max_length - cur_len):
                # Get logits for next token
                logits = self(input_ids)[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample from the filtered distribution
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append next token to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class TrainingLogger:
    """Simple training logger to track metrics"""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Create log file
        self.log_file = self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics for current step"""
        metrics['step'] = step

        # Update internal lists
        if 'train_loss' in metrics:
            self.train_losses.append(metrics['train_loss'])
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'])
        if 'learning_rate' in metrics:
            self.learning_rates.append(metrics['learning_rate'])

        # Save to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def plot_metrics(self):
        """Plot training metrics"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Plot learning rate
        ax2.plot(self.learning_rates, label='Learning Rate')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')

        # Save plot
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png')
        plt.close()


class TextDataset(Dataset):
    """Dataset for language modeling"""

    def __init__(self, texts: List[str], config: Config):
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.data.tokenizer_name)

        # Define special tokens (using common GPT-2 values)
        self.pad_token_id = 50256  # Using the maximum vocab index as pad token

        # Tokenize all texts
        logger.info("Tokenizing texts...")
        self.tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 0:  # Only add non-empty sequences
                self.tokens.extend(tokens)
                self.tokens.append(self.tokenizer.eot_token)  # Add EOT token

        # Create chunks of max_length tokens
        self.chunks = [
            self.tokens[i:i + config.data.max_length]
            for i in range(0, len(self.tokens), config.data.max_length)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # Pad sequence if necessary
        if len(chunk) < self.config.data.max_length:
            chunk = chunk + [self.pad_token_id] * (self.config.data.max_length - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


def create_dataloaders(
    config: Config,
    rank: Optional[int] = None,
    world_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders
    
    Args:
        config: Configuration object
        rank: Process rank for distributed training
        world_size: Total number of processes for distributed training
    """
    # Read text file
    try:
        with open(config.data.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(config.data.data_path, 'r', encoding='latin-1') as f:
            text = f.read()

    # Split into train and validation
    lines = [line for line in text.split('\n') if line.strip()]

    # Apply sample size limit if specified
    if config.data.sample_size:
        lines = lines[:config.data.sample_size]

    # Shuffle and split
    np.random.seed(config.data.random_seed)
    np.random.shuffle(lines)

    split_idx = int(len(lines) * config.data.train_size)
    train_texts = lines[:split_idx]
    val_texts = lines[split_idx:]

    if rank is None or rank == 0:
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")

    # Create datasets
    train_dataset = TextDataset(train_texts, config)
    val_dataset = TextDataset(val_texts, config)

    if rank is not None and world_size is not None:
        # Distributed training mode
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_shuffle = False  # Don't shuffle here as DistributedSampler will handle it
    else:
        # Single GPU mode
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=train_shuffle if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )

    return train_loader, val_loader

def train_epoch(
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Config,
        epoch: int,
        logger: TrainingLogger
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0

    # Setup progress bar
    progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

    for batch_idx, (x, y) in enumerate(progress):
        # Move data to device
        x, y = x.to(config.device), y.to(config.device)

        # Forward pass
        loss = model(x, labels=y)
        loss = loss / config.training.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights if gradient accumulation steps reached
        if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.clip_value
            )

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Update metrics
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]

        # Update progress bar
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{torch.exp(loss).item():.2f}',
            'lr': f'{current_lr:.2e}'
        })

        # Log metrics
        logger.log_metrics({
            'train_loss': loss.item(),
            'learning_rate': current_lr,
            'epoch': epoch,
            'batch': batch_idx
        }, step=epoch * len(train_loader) + batch_idx)

    return total_loss / len(train_loader)


def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Config,
        epoch: int,
        val_loss: float,
        output_dir: Path,
        is_best: bool = False
):
    """Save model checkpoint with more metadata"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'data': {k: str(v) if isinstance(v, Path) else v
                     for k, v in config.data.__dict__.items()}
        },
        'val_loss': val_loss,
    }

    # Save regular checkpoint
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save latest checkpoint for easy loading
    latest_path = output_dir / 'latest_model.pt'
    torch.save(checkpoint, latest_path)
    logger.info(f"Saved latest checkpoint: {latest_path}")

    # Save best model if applicable
    if is_best:
        best_path = output_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model: {best_path}")


def create_optimizer(model: nn.Module, config: Config) -> Tuple[
    torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and scheduler with correct parameter handling"""
    # Get the actual model if it's wrapped in DDP
    actual_model = model.module if isinstance(model, DDP) else model
    
    # Prepare parameter groups
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Embedding)
    
    # Collect all parameter names
    for mn, m in actual_model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f'{mn}.{pn}' if mn else pn
            
            # Skip lambda functions (like our lm_head)
            if isinstance(m, type(lambda: None)):
                continue
            
            # Handle LayerNorm parameters
            if isinstance(m, (nn.LayerNorm, LayerNorm)) or 'norm' in mn.split('.'):
                no_decay.add(fpn)
            # Handle biases
            elif pn.endswith('bias'):
                no_decay.add(fpn)
            # Handle linear and embedding weights
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            # Add remaining parameters to no_decay
            else:
                no_decay.add(fpn)
    
    # Create param_dict based on the actual model
    param_dict = {pn: p for pn, p in actual_model.named_parameters()}
    
    # Validate parameters
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets"
    
    # Create optimizer groups
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": config.training.weight_decay,
            "name": "decay"
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
            "name": "no_decay"
        }
    ]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.training.learning_rate,
        betas=(0.9, 0.95)
    )
    
    # Calculate number of training steps
    if hasattr(config.data, 'sample_size') and config.data.sample_size:
        n_samples = config.data.sample_size
    else:
        # Use a more reliable way to get dataset size
        n_samples = 100000  # Default value, adjust based on your dataset
        try:
            # Try to get the vocab size from the model
            if hasattr(actual_model, 'token_embedding'):
                n_samples = len(actual_model.token_embedding.weight)
        except:
            pass
    
    train_steps_per_epoch = n_samples // config.training.batch_size
    total_steps = config.training.epochs * train_steps_per_epoch
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler


def get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1
):
    """Create a schedule with linear warmup and linear decay"""

    def lr_lambda(current_step):
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
        last_epoch
    )

@torch.no_grad()
def evaluate(
        model: nn.Module,
        val_loader: DataLoader,
        config: Config,
        logger: TrainingLogger,
        step: int
) -> float:
    """Evaluate the model"""
    model.eval()
    total_loss = 0

    for x, y in tqdm(val_loader, desc='Evaluating'):
        x, y = x.to(config.device), y.to(config.device)
        loss = model(x, labels=y)
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))

    # Log metrics
    logger.log_metrics({
        'val_loss': avg_loss,
        'val_perplexity': perplexity.item()
    }, step=step)

    return avg_loss
