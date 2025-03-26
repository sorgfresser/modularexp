#!/usr/bin/env python
import torch
from argparse import Namespace
import numpy as np
import torch.nn.functional as F
import json
import random
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

def perturb_input(input_tuple):
    idx = random.choice([0, 1, 2])
    perturbed = list(input_tuple)
    perturbed[idx] += 1
    return tuple(perturbed)

def run_full_logits(x_tensor, lengths, modules, max_len=50):
    with torch.no_grad():
        encoded = modules["encoder"]("fwd", x=x_tensor, lengths=lengths, causal=False)
        encoded_for_decoding = encoded.transpose(0, 1)
        bs = lengths.size(0)
        # Start with a token: here we use the EOS token as the initial input.
        generated = torch.full((1, bs), modules["decoder"].eos_index, dtype=torch.long, device=x_tensor.device)
        logits_list = []
        for t in range(max_len):
            gen_len = torch.tensor([generated.size(0)], dtype=torch.long, device=x_tensor.device)
            hidden = modules["decoder"].fwd(
                x=generated,
                lengths=gen_len,
                causal=True,
                src_enc=encoded_for_decoding,
                src_len=lengths
            )
            logits_t = modules["decoder"].proj(hidden[-1])  # shape: (bs, n_words)
            logits_list.append(logits_t.unsqueeze(0))  # add a time dimension
            # Greedy decode: choose the argmax token.
            next_token = torch.argmax(logits_t, dim=-1, keepdim=True)  # shape: (bs, 1)
            generated = torch.cat([generated, next_token.transpose(0, 1)], dim=0)
            # If all sequences produced the EOS token, break early.
            if (next_token == modules["decoder"].eos_index).all():
                break
        logits_seq = torch.cat(logits_list, dim=0)  # shape: (T, bs, n_words)
    # Transpose generated to shape (bs, T+1) and return.
    return logits_seq, generated.transpose(0, 1)

def run_full_logits_with_patch(x_tensor, lengths, modules, layer_idx, head_idx, context2_reshaped, max_len=50):
    bs = lengths.size(0)
    candidate_module = modules["decoder"].layers[layer_idx].self_attention

    def patch_hook(module, inp, outp, head_idx=head_idx):
        # inp[0]: original context, shape (bs, qlen, dim)
        context = inp[0]
        bs_local, qlen, dim = context.shape
        n_heads = candidate_module.n_heads
        dim_per_head = dim // n_heads
        # Reshape to (bs, qlen, n_heads, dim_per_head)
        context_reshaped = context.view(bs_local, qlen, n_heads, dim_per_head)
        # Replace only for positions that exist in context2_reshaped.
        replace_length = min(qlen, context2_reshaped.shape[1])
        context_reshaped[:, :replace_length, head_idx, :] = context2_reshaped[:, :replace_length, head_idx, :]
        patched_context = context_reshaped.view(bs_local, qlen, dim)
        return patched_context

    hook_handle = candidate_module.out_lin.register_forward_hook(patch_hook)
    logits_seq, generated_seq = run_full_logits(x_tensor, lengths, modules, max_len)
    hook_handle.remove()
    return logits_seq, generated_seq

def compute_kl_div_full(logits_full, logits_patched):
    T_full = logits_full.shape[0]
    T_patched = logits_patched.shape[0]
    T = min(T_full, T_patched)
    kl_sum = 0.0
    for t in range(T):
        p_full = F.softmax(logits_full[t], dim=-1)
        log_p_patched = F.log_softmax(logits_patched[t], dim=-1)
        kl_sum += F.kl_div(log_p_patched, p_full, reduction='batchmean')
    kl_avg = kl_sum / T
    return kl_avg.item()

def get_layer_context(x_tensor, lengths, modules, layer_idx):
    context_dict = {}
    candidate_module = modules["decoder"].layers[layer_idx].self_attention

    def capture_hook(module, inp, outp):
        # Capture the context (shape: (bs, qlen, dim))
        context_dict["context"] = inp[0].detach().clone()

    hook_handle = candidate_module.out_lin.register_forward_hook(capture_hook)
    with torch.no_grad():
        encoded = modules["encoder"]("fwd", x=x_tensor, lengths=lengths, causal=False)
        encoded_for_decoding = encoded.transpose(0, 1)
        # Run generation to capture the context.
        _ , _ = modules["decoder"].generate(encoded_for_decoding, lengths, max_len=50, sample_temperature=0.7)
    hook_handle.remove()
    return context_dict["context"]

def main():
    # Hyperparameter: KL divergence threshold (τ)
    tau = 0.1

    # Load checkpoint and build modules.
    checkpoint_path = "modularexp/checkpoints/checkpoint (2).pth"
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

    # Input 1: The test input.
    manual_inp1 = (2, 3, 7)
    manual_out1 = fast_exp(*manual_inp1)
    # Input 2: Counterfactual input (randomly perturb one of a, b, or c by 1)
    manual_inp2 = perturb_input(manual_inp1)
    manual_out2 = fast_exp(*manual_inp2)
    print("Correct answer (fast_exp):", manual_out1)
    print("Counterfactual input answer (fast_exp):", manual_out2)
    print("Manual input 1:", manual_inp1)
    print("Manual input 2 (perturbed):", manual_inp2)

    x1 = env.input_encoder.encode(manual_inp1)
    x2 = env.input_encoder.encode(manual_inp2)
    try:
        x1_indices = [env.word2id[token] for token in x1] if isinstance(x1[0], str) else x1
        x2_indices = [env.word2id[token] for token in x2] if isinstance(x2[0], str) else x2
    except KeyError as e:
        print(f"Token not found in vocabulary: {e}")
        return

    x1_tensor = torch.tensor(x1_indices, dtype=torch.long, device=device).unsqueeze(1)
    x2_tensor = torch.tensor(x2_indices, dtype=torch.long, device=device).unsqueeze(1)
    lengths1 = torch.tensor([len(x1_indices)], dtype=torch.long, device=device)
    lengths2 = torch.tensor([len(x2_indices)], dtype=torch.long, device=device)

    # Full model output for input1 using full decoding.
    logits_full_seq, generated_full_seq = run_full_logits(x1_tensor, lengths1, modules, max_len=50)
    generated_indices_full = generated_full_seq[0].tolist()
    decoded_full = idx_to_infix(env, generated_indices_full, input=False)
    print("\nFull model output (unpatched):")
    print(decoded_full)

    # ACDC: Iterate over each decoder layer and each attention head.
    final_circuit = {}
    num_layers = len(modules["decoder"].layers)
    # For ACDC, we iterate from later to earlier layers.
    for layer_idx in reversed(range(num_layers)):
        candidate_module = modules["decoder"].layers[layer_idx].self_attention
        n_heads = candidate_module.n_heads
        dim = candidate_module.dim
        dim_per_head = dim // n_heads
        print(f"\n--- Processing decoder layer {layer_idx} ---")
        # Get context from the counterfactual input for this layer.
        context2 = get_layer_context(x2_tensor, lengths2, modules, layer_idx)
        bs, qlen, _ = context2.shape
        # Reshape context2 to (bs, qlen, n_heads, dim_per_head)
        context2_reshaped = context2.view(bs, qlen, n_heads, dim_per_head)
        final_circuit[layer_idx] = []
        # Iterate over heads (from higher- to lower-indexed).
        for head_idx in reversed(range(n_heads)):
            logits_patched_seq, generated_patched_seq = run_full_logits_with_patch(
                x1_tensor, lengths1, modules, layer_idx, head_idx, context2_reshaped, max_len=50
            )
            kl_div = compute_kl_div_full(logits_full_seq, logits_patched_seq)
            generated_indices_patched = generated_patched_seq[0].tolist()
            decoded_patched = idx_to_infix(env, generated_indices_patched, input=False)

            print(f"Layer {layer_idx} head {head_idx}: KL divergence = {kl_div:.4f}")
            print("Patched output:", decoded_patched)
            # ACDC criterion: if patching causes a significant change (KL > tau),
            # then this head is critical and is kept in the circuit.
            if kl_div > tau:
                final_circuit[layer_idx].append(head_idx)
        print(f"Critical heads in layer {layer_idx}: {final_circuit[layer_idx]}")

    print("\n=== Final Stored Circuit ===")
    print(final_circuit)

    with open("final_circuit.json", "w") as f:
        json.dump(final_circuit, f, indent=4)
    print("Final circuit saved to final_circuit.json")

if __name__ == '__main__':
    main()
