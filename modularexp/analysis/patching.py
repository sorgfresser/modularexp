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
    # Load checkpoint and build modules
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

    #########################################
    # Define two manual inputs and their expected outputs
    a1, b1, c1 = 200, 30, 7   # Input 1
    a2, b2, c2 = 123, 45, 7   # Input 2

    manual_inp1 = (a1, b1, c1)
    manual_out1 = fast_exp(a1, b1, c1)
    print("Input 1 (input, solution):", manual_inp1 + (manual_out1,))

    manual_inp2 = (a2, b2, c2)
    manual_out2 = fast_exp(a2, b2, c2)
    print("Input 2 (input, solution):", manual_inp2 + (manual_out2,))

    #########################################
    # Encode inputs using environment encoder
    x1 = env.input_encoder.encode(manual_inp1)
    x2 = env.input_encoder.encode(manual_inp2)

    # Convert tokens to indices and add the EOS token (as seen in training)
    if isinstance(x1[0], str):
        try:
            x1_indices = [env.word2id[token] for token in x1] + [env.eos_index]
            x2_indices = [env.word2id[token] for token in x2] + [env.eos_index]
        except KeyError as e:
            print(f"Token not found in vocabulary: {e}")
            return
    else:
        x1_indices = x1 + [env.eos_index]
        x2_indices = x2 + [env.eos_index]

    print("\nInput 1 indices:", x1_indices)
    print("Input 2 indices:", x2_indices)

    # The model expects (sequence_length, batch_size); here batch_size = 1.
    x1_tensor = torch.tensor(x1_indices, dtype=torch.long, device=device).unsqueeze(1)
    x2_tensor = torch.tensor(x2_indices, dtype=torch.long, device=device).unsqueeze(1)
    lengths1 = torch.tensor([len(x1_indices)], dtype=torch.long, device=device)
    lengths2 = torch.tensor([len(x2_indices)], dtype=torch.long, device=device)

    #########################################
    # Generate expected outputs for both inputs (without patching)
    with torch.no_grad():
        # Input 1 generation
        encoded1 = modules["encoder"]("fwd", x=x1_tensor, lengths=lengths1, causal=False)
        encoded_for_decoding1 = encoded1.transpose(0, 1)
        generated1, gen_lengths1 = modules["decoder"].generate(
            encoded_for_decoding1, lengths1, max_len=50, sample_temperature=0.7
        )
        # Input 2 generation
        encoded2 = modules["encoder"]("fwd", x=x2_tensor, lengths=lengths2, causal=False)
        encoded_for_decoding2 = encoded2.transpose(0, 1)
        generated2, gen_lengths2 = modules["decoder"].generate(
            encoded_for_decoding2, lengths2, max_len=50, sample_temperature=0.7
        )

    decoded_expression1 = idx_to_infix(env, generated1[:, 0].tolist(), input=False)
    decoded_expression2 = idx_to_infix(env, generated2[:, 0].tolist(), input=False)
    print("\nExpected output for Input 1 (decoded):")
    print(decoded_expression1)
    print("\nExpected output for Input 2 (decoded):")
    print(decoded_expression2)

    #########################################
    # ACTIVATION PATCHING EXPERIMENT
    #
    # Choose a candidate module from the decoder.
    # Here we use the FFN of the first decoder layer.
    candidate_module = modules["decoder"].layers[0].ffn

    activations = {}

    def save_activation(module, input, output):
        # Save a clone of the activation
        activations["candidate"] = output.detach().clone()

    # Register a hook to capture activations during the forward pass.
    hook_handle = candidate_module.register_forward_hook(save_activation)

    # Run the model on Input 1 to capture its candidate activation.
    with torch.no_grad():
        encoded1_patch = modules["encoder"]("fwd", x=x1_tensor, lengths=lengths1, causal=False)
        encoded_for_decoding1_patch = encoded1_patch.transpose(0, 1)
        _ , _ = modules["decoder"].generate(
            encoded_for_decoding1_patch, lengths1, max_len=50, sample_temperature=0.7
        )
    activation_input1 = activations["candidate"].clone()
    activations.clear()

    # Run the model on Input 2 to capture its candidate activation.
    with torch.no_grad():
        encoded2_patch = modules["encoder"]("fwd", x=x2_tensor, lengths=lengths2, causal=False)
        encoded_for_decoding2_patch = encoded2_patch.transpose(0, 1)
        _ , _ = modules["decoder"].generate(
            encoded_for_decoding2_patch, lengths2, max_len=50, sample_temperature=0.7
        )
    activation_input2 = activations["candidate"].clone()

    # Remove the original hook since we now have both activations.
    hook_handle.remove()

    # Now, patch the candidate activation: when processing Input 1, replace its activation with that from Input 2.
    def patch_activation(module, input, output):
        return activation_input2  # Use activation from Input 2

    patch_handle = candidate_module.register_forward_hook(patch_activation)

    with torch.no_grad():
        encoded1_patched = modules["encoder"]("fwd", x=x1_tensor, lengths=lengths1, causal=False)
        encoded_for_decoding1_patched = encoded1_patched.transpose(0, 1)
        generated_patched, _ = modules["decoder"].generate(
            encoded_for_decoding1_patched, lengths1, max_len=50, sample_temperature=0.7
        )
    patch_handle.remove()

    decoded_expression_patched = idx_to_infix(env, generated_patched[:, 0].tolist(), input=False)
    print("\nPatched output (Input 1 with candidate activation from Input 2):")
    print(decoded_expression_patched)

if __name__ == '__main__':
    main()
