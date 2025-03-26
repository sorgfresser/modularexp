#!/usr/bin/env python
import torch
from argparse import Namespace
import numpy as np
from modularexp.envs import build_env
from modularexp.model import build_modules

def fast_exp(b, e, m):
    r = 1
    if e & 1:
        r = b % m 
    while e:
        e >>= 1
        b = (b * b) % m
        if e & 1:
            r = (r * b) % m
    return r

def idx_to_infix(env, idx, input=True):
    prefix = [env.id2word[wid] for wid in idx]
    return env.input_to_infix(prefix) if input else env.output_to_infix(prefix)

def main():
    checkpoint_path = "modularexp/checkpoints/checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    params = Namespace(**checkpoint["params"])
    if isinstance(params.tasks, list):
        params.tasks = ",".join(params.tasks)
    
    env = build_env(params)
    if not hasattr(env, "rng"):
        seed = params.env_base_seed if hasattr(params, "env_base_seed") else 42
        env.rng = np.random.RandomState(seed)
    
    modules = build_modules(env, params)
    for k, v in modules.items():
        v.load_state_dict(checkpoint[k])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for key, module in modules.items():
        module.to(device)
        module.eval()
    
    # Manual input
    a, b, c = 200, 30, 7
    manual_inp = (a, b, c)
    manual_out = fast_exp(a, b, c)
    print("Manually generated expression (input, solution):")
    print(manual_inp + (manual_out,))
    
    x = env.input_encoder.encode(manual_inp)
    y = env.output_encoder.encode(manual_out)
    print("\nEncoded input tokens:")
    print(x)
    print("\nEncoded output tokens (target):")
    print(y)
    
    if isinstance(x[0], str):
        try:
            x_indices = [env.word2id[token] for token in x] + [env.eos_index]
        except KeyError as e:
            print(f"Token not found in vocabulary: {e}")
            return
    else:
        x_indices = x
    print("\nInput indices:")
    print(x_indices)
    
    # The model expects (sequence_length, batch_size); here batch_size=1
    x_tensor = torch.tensor(x_indices, dtype=torch.long, device=device).unsqueeze(1)
    lengths = torch.tensor([len(x_indices)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        encoded = modules["encoder"]("fwd", x=x_tensor, lengths=lengths, causal=False)
        encoded_for_decoding = encoded.transpose(0, 1)
        generated, gen_lengths = modules["decoder"].generate(
            encoded_for_decoding, lengths, max_len=50, sample_temperature=0.7
        )
        # generated = generated[1:]
    
    print("\nGenerated output tensor (token indices):")
    print(generated)
    
    # Assume batch size is 1; get the sequence as a list
    generated_indices = generated[:, 0].tolist()
    decoded_expression = idx_to_infix(env, generated_indices, input=False)
    print("\nDecoded output expression (evaluator style):")
    print(decoded_expression)

if __name__ == '__main__':
    main()
