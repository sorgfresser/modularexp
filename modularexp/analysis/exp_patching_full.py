#!/usr/bin/env python
import torch
from argparse import Namespace
import numpy as np
import torch.nn.functional as F
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from modularexp.envs import build_env
from modularexp.model import build_modules

# ----------------------------
# Utility and Decoding Methods
# ----------------------------

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

def generate_sample():
    a = random.randint(2, 5)
    b = random.randint(2, 4)
    base = a ** b
    c = random.randint(base + 1, base + 10)
    full_inp = (a, b, c)
    counterfactual_inp = perturb_input(full_inp)
    return full_inp, counterfactual_inp

def generate_samples(n_samples):
    samples = []
    while len(samples) < n_samples:
        full_inp, cf_inp = generate_sample()
        a, b, c = full_inp
        if c > a ** b:
            samples.append((full_inp, cf_inp))
    return samples

def run_full_logits(x_tensor, lengths, modules, max_len=50):
    with torch.no_grad():
        encoded = modules["encoder"]("fwd", x=x_tensor, lengths=lengths, causal=False)
        encoded_for_decoding = encoded.transpose(0, 1)
        bs = lengths.size(0)
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
            logits_t = modules["decoder"].proj(hidden[-1])
            logits_list.append(logits_t.unsqueeze(0))
            next_token = torch.argmax(logits_t, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.transpose(0, 1)], dim=0)
            if (next_token == modules["decoder"].eos_index).all():
                break
        logits_seq = torch.cat(logits_list, dim=0)
    return logits_seq, generated.transpose(0, 1)

def run_full_logits_with_patch(x_tensor, lengths, modules, layer_idx, head_idx, context2_reshaped, max_len=50):
    bs = lengths.size(0)
    candidate_module = modules["decoder"].layers[layer_idx].self_attention
    def patch_hook(module, inp, outp, head_idx=head_idx):
        context = inp[0]
        bs_local, qlen, dim = context.shape
        n_heads = candidate_module.n_heads
        dim_per_head = dim // n_heads
        context_reshaped = context.view(bs_local, qlen, n_heads, dim_per_head)
        replace_length = min(qlen, context2_reshaped.shape[1])
        context_reshaped[:, :replace_length, head_idx, :] = context2_reshaped[:, :replace_length, head_idx, :]
        patched_context = context_reshaped.view(bs_local, qlen, dim)
        return patched_context
    hook_handle = candidate_module.out_lin.register_forward_hook(patch_hook)
    logits_seq, generated_seq = run_full_logits(x_tensor, lengths, modules, max_len)
    hook_handle.remove()
    return logits_seq, generated_seq

def run_full_logits_with_circuit(x_tensor, lengths, modules, circuit, cf_contexts, max_len=50):
    hook_handles = []
    for layer_idx, head_list in circuit.items():
        if not head_list:
            continue
        candidate_module = modules["decoder"].layers[layer_idx].self_attention
        context2_reshaped = cf_contexts[layer_idx]
        def make_patch_hook(head_list, context2_reshaped):
            def patch_hook(module, inp, outp):
                context = inp[0]
                bs_local, qlen, dim = context.shape
                n_heads = candidate_module.n_heads
                dim_per_head = dim // n_heads
                context_reshaped = context.view(bs_local, qlen, n_heads, dim_per_head)
                replace_length = min(qlen, context2_reshaped.shape[1])
                for head_idx in head_list:
                    context_reshaped[:, :replace_length, head_idx, :] = context2_reshaped[:, :replace_length, head_idx, :]
                patched_context = context_reshaped.view(bs_local, qlen, dim)
                return patched_context
            return patch_hook
        hook = candidate_module.out_lin.register_forward_hook(
            make_patch_hook(head_list, context2_reshaped)
        )
        hook_handles.append((candidate_module, hook))
    logits_seq, generated_seq = run_full_logits(x_tensor, lengths, modules, max_len)
    for module, hook in hook_handles:
        hook.remove()
    return logits_seq, generated_seq

# ----------------------------
# Evaluation Functions (using numerical comparison)
# ----------------------------

def check_hypothesis_numeric(eq, env, target_val):
    # Convert predicted indices to tokens.
    pred_tokens = [env.id2word[token] for token in eq["hyp"]]
    # Filter out EOS tokens.
    filtered = [tok for tok in pred_tokens if tok != "<eos>" and tok != env.id2word[env.eos_index]]
    # Get the correct sign from the target tokens.
    target_tokens = env.output_encoder.encode(target_val)
    correct_sign = target_tokens[0]
    if not filtered or filtered[0] not in ['+', '-']:
        # Force in the correct sign.
        if filtered:
            filtered[0] = correct_sign
        else:
            filtered = [correct_sign]
    pred_val = env.output_encoder.decode(filtered)
    eq["decoded_pred"] = pred_val
    eq["decoded_tgt"] = target_val
    eq["is_valid"] = 1 if pred_val == target_val else 0
    eq["filtered_pred_tokens"] = filtered
    return eq

def evaluate_full_accuracy(samples, modules, env, device, max_len=50):
    correct = 0
    evaluations = []
    example_printed = 0
    for full_inp, _ in tqdm(samples, desc="Evaluating full model accuracy"):
        a, b, c = full_inp
        x = env.input_encoder.encode(full_inp)
        try:
            x_indices = [env.word2id[token] for token in x] if isinstance(x[0], str) else x
            x_indices = x_indices + [env.eos_index]
        except KeyError as e:
            print(f"Token error: {e}")
            continue
        x_tensor = torch.tensor(x_indices, dtype=torch.long, device=device).unsqueeze(1)
        lengths = torch.tensor([len(x_indices)], dtype=torch.long, device=device)
        _, generated_seq = run_full_logits(x_tensor, lengths, modules, max_len)
        pred_indices = generated_seq[0].tolist()[1:]
        target_tokens = env.output_encoder.encode(a ** b)
        target = [env.word2id[token] for token in target_tokens] + [env.eos_index]
        eq = {"src": x_indices, "tgt": target, "hyp": pred_indices}
        eq = check_hypothesis_numeric(eq, env, a ** b)
        if eq["is_valid"] > 0:
            correct += 1
        if example_printed < 5:
            print("\n--- Full Model Example ---")
            print("Input:", full_inp)
            print("Target (numerical):", a ** b)
            print("Decoded Target:", eq["decoded_tgt"])
            print("Decoded Prediction:", eq["decoded_pred"])
            print("Token target:", target_tokens + ["<eos>"])
            print("Token prediction:", [env.id2word[i] for i in eq["hyp"]])
            print("Filtered predicted tokens:", eq["filtered_pred_tokens"])
            example_printed += 1
        evaluations.append(eq)
    accuracy = correct / len(samples) if samples else 0.0
    return accuracy, evaluations

def evaluate_circuit_accuracy(samples, modules, env, device, circuit, max_len=50):
    correct = 0
    evaluations = []
    example_printed = 0
    for full_inp, cf_inp in tqdm(samples, desc="Evaluating circuit accuracy"):
        a, b, c = full_inp
        x = env.input_encoder.encode(full_inp)
        try:
            x_indices = [env.word2id[token] for token in x] if isinstance(x[0], str) else x
            x_indices = x_indices + [env.eos_index]
        except KeyError as e:
            print(f"Token error: {e}")
            continue
        x_tensor = torch.tensor(x_indices, dtype=torch.long, device=device).unsqueeze(1)
        lengths = torch.tensor([len(x_indices)], dtype=torch.long, device=device)
        cf = env.input_encoder.encode(cf_inp)
        try:
            cf_indices = [env.word2id[token] for token in cf] if isinstance(cf[0], str) else cf
        except KeyError as e:
            print(f"Token error: {e}")
            continue
        cf_tensor = torch.tensor(cf_indices, dtype=torch.long, device=device).unsqueeze(1)
        lengths_cf = torch.tensor([len(cf_indices)], dtype=torch.long, device=device)
        num_layers = len(modules["decoder"].layers)
        cf_contexts = {}
        for layer_idx in range(num_layers):
            if layer_idx in circuit and circuit[layer_idx]:
                context = get_layer_context(cf_tensor, lengths_cf, modules, layer_idx)
                bs, qlen, dim = context.shape
                candidate_module = modules["decoder"].layers[layer_idx].self_attention
                n_heads = candidate_module.n_heads
                dim_per_head = dim // n_heads
                cf_contexts[layer_idx] = context.view(bs, qlen, n_heads, dim_per_head)
        _, generated_seq = run_full_logits_with_circuit(x_tensor, lengths, modules, circuit, cf_contexts, max_len)
        pred_indices = generated_seq[0].tolist()[1:]
        target_tokens = env.output_encoder.encode(a ** b)
        target = [env.word2id[token] for token in target_tokens] + [env.eos_index]
        eq = {"src": x_indices, "tgt": target, "hyp": pred_indices}
        eq = check_hypothesis_numeric(eq, env, a ** b)
        if eq["is_valid"] > 0:
            correct += 1
        if example_printed < 5:
            print("\n--- Circuit Model Example ---")
            print("Input:", full_inp)
            print("Target (numerical):", a ** b)
            print("Decoded Target:", eq["decoded_tgt"])
            print("Decoded Prediction:", eq["decoded_pred"])
            print("Token target:", target_tokens + ["<eos>"])
            print("Token prediction:", [env.id2word[i] for i in eq["hyp"]])
            print("Filtered predicted tokens:", eq["filtered_pred_tokens"])
            example_printed += 1
        evaluations.append(eq)
    accuracy = correct / len(samples) if samples else 0.0
    return accuracy, evaluations

def get_layer_context(x_tensor, lengths, modules, layer_idx):
    context_dict = {}
    candidate_module = modules["decoder"].layers[layer_idx].self_attention
    def capture_hook(module, inp, outp):
        context_dict["context"] = inp[0].detach().clone()
    hook_handle = candidate_module.out_lin.register_forward_hook(capture_hook)
    with torch.no_grad():
        encoded = modules["encoder"]("fwd", x=x_tensor, lengths=lengths, causal=False)
        encoded_for_decoding = encoded.transpose(0, 1)
        _, _ = modules["decoder"].generate(encoded_for_decoding, lengths, max_len=50, sample_temperature=0.7)
    hook_handle.remove()
    return context_dict["context"]

# ----------------------------
# Main Script
# ----------------------------

def main():
    tau = 0.1
    n_samples = 1000
    max_len = 50
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
    for module in modules.values():
        module.to(device)
        module.eval()
    samples = generate_samples(n_samples)
    print(f"Generated {len(samples)} samples satisfying c > a**b.")
    full_acc, full_evals = evaluate_full_accuracy(samples, modules, env, device, max_len)
    print(f"\nFull model accuracy (numerical equivalence): {full_acc*100:.2f}%")
    num_layers = len(modules["decoder"].layers)
    n_heads = modules["decoder"].layers[0].self_attention.n_heads
    kl_accum = {layer: {head: 0.0 for head in range(n_heads)} for layer in range(num_layers)}
    count = 0
    for full_inp, cf_inp in tqdm(samples, desc="Computing KL divergences"):
        x = env.input_encoder.encode(full_inp)
        cf = env.input_encoder.encode(cf_inp)
        try:
            x_indices = [env.word2id[token] for token in x] if isinstance(x[0], str) else x
            x_indices = x_indices + [env.eos_index]
            cf_indices = [env.word2id[token] for token in cf] if isinstance(cf[0], str) else cf
        except KeyError as e:
            print(f"Token error: {e}")
            continue
        x_tensor = torch.tensor(x_indices, dtype=torch.long, device=device).unsqueeze(1)
        lengths = torch.tensor([len(x_indices)], dtype=torch.long, device=device)
        cf_tensor = torch.tensor(cf_indices, dtype=torch.long, device=device).unsqueeze(1)
        lengths_cf = torch.tensor([len(cf_indices)], dtype=torch.long, device=device)
        for layer_idx in range(num_layers):
            candidate_module = modules["decoder"].layers[layer_idx].self_attention
            context_cf = get_layer_context(cf_tensor, lengths_cf, modules, layer_idx)
            bs, qlen, dim = context_cf.shape
            dim_per_head = dim // candidate_module.n_heads
            context_cf_reshaped = context_cf.view(bs, qlen, candidate_module.n_heads, dim_per_head)
            for head_idx in range(n_heads):
                logits_full_seq, _ = run_full_logits(x_tensor, lengths, modules, max_len)
                logits_patched_seq, _ = run_full_logits_with_patch(x_tensor, lengths, modules,
                                                                    layer_idx, head_idx, context_cf_reshaped, max_len)
                T_full = logits_full_seq.shape[0]
                T_patched = logits_patched_seq.shape[0]
                T = min(T_full, T_patched)
                kl_sum = 0.0
                for t in range(T):
                    p_full = F.softmax(logits_full_seq[t], dim=-1)
                    log_p_patched = F.log_softmax(logits_patched_seq[t], dim=-1)
                    kl_sum += F.kl_div(log_p_patched, p_full, reduction='batchmean')
                kl_avg = kl_sum / T
                kl_accum[layer_idx][head_idx] += kl_avg.item()
        count += 1
    avg_kl = {layer: {head: kl_accum[layer][head] / count for head in range(n_heads)} for layer in range(num_layers)}
    final_circuit = {}
    for layer in range(num_layers):
        final_circuit[layer] = [head for head in range(n_heads) if avg_kl[layer][head] > tau]
    print("\n=== Final Circuit (averaged over samples) ===")
    print(final_circuit)
    with open("final_circuit.json", "w") as f:
        json.dump(final_circuit, f, indent=4)
    print("Final circuit saved to final_circuit.json")
    circuit_acc, circuit_evals = evaluate_circuit_accuracy(samples, modules, env, device, final_circuit, max_len)
    print(f"\nCircuit accuracy (numerical equivalence): {circuit_acc*100:.2f}%")
    kl_matrix = np.zeros((num_layers, n_heads))
    for layer in range(num_layers):
        for head in range(n_heads):
            kl_matrix[layer, head] = avg_kl[layer][head]
    plt.figure(figsize=(8, 6))
    plt.imshow(kl_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label="Average KL Divergence")
    plt.xlabel("Head Index")
    plt.ylabel("Decoder Layer")
    plt.title("Average KL Divergence per Head Across Samples")
    plt.xticks(range(n_heads))
    plt.yticks(range(num_layers))
    plt.savefig("kl_divergence_heatmap.png")
    plt.show()
    print("KL divergence heatmap saved as 'kl_divergence_heatmap.png'.")

if __name__ == '__main__':
    main()
