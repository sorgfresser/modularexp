#!/usr/bin/env python
import torch
from argparse import Namespace
import numpy as np
from modularexp.envs import build_env
from modularexp.model import build_modules

def main():
    # Path 
    checkpoint_path = "./modularexp/checkpoints/checkpoint.pth"
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = Namespace(**checkpoint["params"])
    
    if isinstance(params.tasks, list):
        params.tasks = ','.join(params.tasks)
    
    # Build the environment
    env = build_env(params)
    # Initialize the RNG if needed (some parts of the code expect env.rng)
    if not hasattr(env, "rng"):
        seed = params.env_base_seed if hasattr(params, "env_base_seed") else 42
        env.rng = np.random.RandomState(seed)
    
    # Build model modules
    modules = build_modules(env, params)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for key, module in modules.items():
        module.to(device)
        module.eval()
    
    manual_input = [2, 3, 5]  # base=2, exponent=3, modulus=5
    
    x = env.input_encoder.encode(manual_input)
    
    print("Manually set input tokens:")
    print(x)
    
    # If the tokens are strings, convert them to indices using env.word2id.
    if isinstance(x[0], str):
        x_indices = [env.word2id[token] for token in x]
    else:
        x_indices = x

    x_tensor = torch.tensor(x_indices, dtype=torch.long, device=device).unsqueeze(1)
    lengths = torch.tensor([len(x_indices)], dtype=torch.long, device=device)
    
    # Forward pass through the encoder and then use the decoder's generate method
    with torch.no_grad():
        encoded = modules["encoder"]("fwd", x=x_tensor, lengths=lengths, causal=False)
        encoded_for_decoding = encoded.transpose(0, 1)

        generated, gen_lengths = modules["decoder"].generate(
            encoded_for_decoding, lengths, max_len=50
        )
    
    print("\nGenerated output tensor (token indices):")
    print(generated)
    
    output_tokens = [env.id2word[idx.item()] for idx in generated[:, 0]]
    print("\nGenerated output tokens:")
    print(output_tokens)
    
if __name__ == '__main__':
    main()
