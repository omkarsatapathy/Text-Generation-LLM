import torch
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
import argparse
from pathlib import Path
from config import Config
from utils import GPTDecoder
import logging


# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = None) -> tuple[GPTDecoder, Config]:
    """Load model from checkpoint with proper error handling"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Loading model from {checkpoint_path} to {device}")

    try:
        # Load checkpoint with safety settings
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True  # For safety
        )
        
        # Print checkpoint keys for debugging
        logger.info(f"Checkpoint contains keys: {checkpoint.keys()}")
        
        # Reconstruct config
        config = Config()
        if isinstance(checkpoint['config'], dict):
            # Handle nested dictionary structure
            if 'model' in checkpoint['config']:
                config.model.__dict__.update(checkpoint['config']['model'])
            if 'training' in checkpoint['config']:
                config.training.__dict__.update(checkpoint['config']['training'])
            if 'data' in checkpoint['config']:
                config.data.__dict__.update(checkpoint['config']['data'])
        
        # Log config for debugging
        logger.info(f"Model configuration loaded: d_model={config.model.d_model}, "
                   f"num_layers={config.model.num_layers}, "
                   f"num_head={config.model.num_head}")
        
        # Create and load model
        model = GPTDecoder(config).to(device)
        
        # Load state dict with error checking
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            logger.error(f"Error loading state dict: {e}")
            # Try loading without module prefix if needed
            if 'module' in next(iter(checkpoint['model_state_dict'].keys())):
                # Remove 'module.' prefix for distributed training checkpoints
                new_state_dict = {k[7:]: v for k, v in checkpoint['model_state_dict'].items()}
                model.load_state_dict(new_state_dict)
        
        model.eval()
        
        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully with {total_params:,} parameters")
        
        return model, config
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def generate_text_from_model(
        model: GPTDecoder,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        device: str = None
) -> str:
    """Generate text from the model with improved parameters"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Ensure model is in evaluation mode
    model.eval()

    # Tokenize the prompt
    try:
        # For tiktoken, we don't need special token handling during encoding
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return ""

    # Track generated tokens
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model's output
            try:
                logits = model(input_ids)
                
                # Get the next token logits (using last token's predictions)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Break if we generate EOT token
                if next_token.item() == tokenizer.eot_token:
                    break
                    
                # Append to generated sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                
            except Exception as e:
                print(f"Error during generation step: {e}")
                break
            
            # Break if the sequence gets too long
            if len(generated_tokens) >= max_new_tokens:
                break
    
    try:
        # For tiktoken, just decode normally
        full_text = tokenizer.decode(input_ids[0].tolist())
        return full_text
    except Exception as e:
        print(f"Error during decoding: {e}")
        return prompt  # Return original prompt if decoding fails

def main():
    parser = argparse.ArgumentParser(description='Generate text using trained GPT model')
    parser.add_argument('--model_path', type=str, default='outputs/latest_model.pt',
                      help='Path to model checkpoint')
    parser.add_argument('--max_tokens', type=int, default=100,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                      help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='Top-p (nucleus) sampling parameter')
    args = parser.parse_args()

    # Load model
    try:
        model, config = load_model(args.model_path)
        logger.info("Model loaded successfully!")
        
        # Print some model info
        logger.info(f"Model config: {config.model}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Add validation loss info
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'val_loss' in checkpoint:
            logger.info(f"Validation loss from checkpoint: {checkpoint['val_loss']:.4f}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Print vocabulary size
    logger.info(f"Tokenizer vocabulary size: {tokenizer.n_vocab}")
    if tokenizer.n_vocab != config.model.vocab_size:
        logger.warning(f"Warning: Model vocab size ({config.model.vocab_size}) "
                      f"doesn't match tokenizer vocab size ({tokenizer.n_vocab})")

    # Interactive generation loop
    print("\nEnter 'quit' to exit")
    while True:
        try:
            prompt = input("\nEnter your prompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break

            print("\nGenerating text...\n")
            
            # Track generation time
            start_time = time.time()
            
            generated_text = generate_text_from_model(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            
            generation_time = time.time() - start_time

            print("Generated Text:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)
            print(f"Generation took {generation_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user")
            break
        except Exception as e:
            print(f"Error during generation: {e}")
            continue

if __name__ == "__main__":
    import time  # Add this import at the top of the file
    main()