from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import torch

@dataclass
class ModelConfig:
    d_model: int = 768
    num_head: int = 16
    drop_prob: float = 0.1
    ffn_hidden: int = 3072  # 4x d_model is common for decoder architectures
    num_layers: int = 32
    max_seq_len: int = 1024
    vocab_size: int = 50257  # GPT-2 vocab size (incding pad token)
    use_flash_attention: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 1024
    learning_rate: float = 3e-4
    epochs: int = 15
    warmup_steps: int = 1000
    clip_value: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1


@dataclass
class DataConfig:
    data_path: Path = Path('shakespeare.txt')  # Example dataset
    train_size: float = 0.9
    random_seed: int = 42
    max_length: int = 1024
    sample_size: Optional[int] = None  # Set to None to use full dataset
    tokenizer_name: str = "gpt2"  # Used for tiktoken encoding


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()

    def __post_init__(self):
        # Convert data_path to Path object if it's a string
        if isinstance(self.data.data_path, str):
            self.data.data_path = Path(self.data.data_path)

    @property
    def device(self) -> torch.device:
        return torch.device(self.training.device)

    def save(self, path: str = "config.json"):
        """Save configuration to a JSON file"""
        import json

        config_dict = {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": {k: str(v) if isinstance(v, Path) else v
                     for k, v in self.data.__dict__.items()}
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str = "config.json") -> 'Config':
        """Load configuration from a JSON file"""
        import json

        with open(path) as f:
            config_dict = json.load(f)

        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])
        data_config = DataConfig(**config_dict["data"])

        return cls(
            model=model_config,
            training=training_config,
            data=data_config
        )

    def display(self):
        """Display the current configuration"""
        from pprint import pprint
        print("Current Configuration:")
        print("\nModel Config:")
        pprint(self.model.__dict__)
        print("\nTraining Config:")
        pprint(self.training.__dict__)
        print("\nData Config:")
        pprint(self.data.__dict__)