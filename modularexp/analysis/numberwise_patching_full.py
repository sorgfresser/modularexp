#!/usr/bin/env python
import os
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
import math

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

# ----------------------------
# Sample Generation
# ----------------------------

def generate_sample(mod_main, mod_cf):
    """
    Generate a single sample with:
      - a and b random.
      - full sample: c is a multiple of mod_main and > a**b.
      - counterfactual: c is a multiple of mod_cf and < a**b.
    If a valid counterfactual cannot be found, the sample is rejected.
    """
    while True:
        a = random.randint(2, 5)
        b = random.randint(2, 4)
        base = a ** b
        # For full sample: choose the smallest multiple of mod_main that is > base, with some random offset
        k_full = (base // mod_main) + 1
        k_full += random.randint(0, 5)
        c_full = k_full * mod_main

        # For counterfactual: choose the largest multiple of mod_cf that is < base
        k_cf = (base - 1) // mod_cf  # floor division
        if k_cf < 1:
            continue  # skip samples where a counterfactual multiple isn't available
        c_cf = k_cf * mod_cf

        # Ensure we have a proper ordering
        if c_full > base and c_cf < base:
            return (a, b, c_full), (a, b, c_cf)

def generate_samples(n_samples, mod_main, mod_cf):
    samples = []
    while len(samples) < n_samples:
        sample = generate_sample(mod_main, mod_cf)
        if sample is not None:
            samples.append(sample)
    return samples

# ----------------------------
# Decoding and Logits Functions
# ----------------------------

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
    target_tokens = env.output_encoder.encode(target_val)
    correct_sign = target_tokens[0]
    if not filtered or filtered[0] not in ['+', '-']:
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
# Pipeline Run Function
# ----------------------------

def run_pipeline(run_id, mod_main, mod_cf, n_samples=50, max_len=50):
    # Check if previous results exist (specific to this run_id)
    full_eval_filename = f"{run_id}_full_evals.json"
    circuit_eval_filename = f"{run_id}_circuit_evals.json"
    final_circuit_filename = f"{run_id}_final_circuit.json"
    if os.path.exists(full_eval_filename) and os.path.exists(circuit_eval_filename) and os.path.exists(final_circuit_filename):
        with open(full_eval_filename, "r") as f:
            full_results = json.load(f)
        with open(circuit_eval_filename, "r") as f:
            circuit_results = json.load(f)
        with open(final_circuit_filename, "r") as f:
            final_circuit = json.load(f)
        print(f"Loaded previous evaluation results for run {run_id}:")
        print("Full model accuracy (numerical equivalence): {:.2f}%".format(full_results["accuracy"] * 100))
        print("Circuit accuracy (numerical equivalence): {:.2f}%".format(circuit_results["accuracy"] * 100))
        print("Final circuit:", final_circuit)
    else:
        tau = 0.1
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
        # Generate samples with the desired moduli.
        samples = generate_samples(n_samples, mod_main, mod_cf)
        print(f"Run {run_id}: Generated {len(samples)} samples with full sample c multiple of {mod_main} and counterfactual c multiple of {mod_cf}.")
        # Evaluate full model accuracy.
        full_acc, full_evals = evaluate_full_accuracy(samples, modules, env, device, max_len)
        print(f"\nRun {run_id}: Full model accuracy (numerical equivalence): {full_acc*100:.2f}%")
        with open(full_eval_filename, "w") as f:
            json.dump({"accuracy": full_acc, "evaluations": full_evals}, f, indent=4)
        # Compute KL divergences.
        num_layers = len(modules["decoder"].layers)
        n_heads = modules["decoder"].layers[0].self_attention.n_heads
        kl_accum = {layer: {head: 0.0 for head in range(n_heads)} for layer in range(num_layers)}
        count = 0
        for full_inp, cf_inp in tqdm(samples, desc="Run {}: Computing KL divergences".format(run_id)):
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
        # Determine final circuit.
        final_circuit = {}
        for layer in range(num_layers):
            final_circuit[layer] = [head for head in range(n_heads) if avg_kl[layer][head] > tau]
        print("\n=== Run {}: Final Circuit (averaged over samples) ===".format(run_id))
        print(final_circuit)
        with open(final_circuit_filename, "w") as f:
            json.dump(final_circuit, f, indent=4)
        print(f"Final circuit saved to {final_circuit_filename}")
        circuit_acc, circuit_evals = evaluate_circuit_accuracy(samples, modules, env, device, final_circuit, max_len)
        print(f"\nRun {run_id}: Circuit accuracy (numerical equivalence): {circuit_acc*100:.2f}%")
        with open(circuit_eval_filename, "w") as f:
            json.dump({"accuracy": circuit_acc, "evaluations": circuit_evals}, f, indent=4)
        with open(f"{run_id}_kl_matrix.json", "w") as f:
            json.dump(avg_kl, f, indent=4)
        # Plot KL divergence heatmap.
        kl_matrix = np.zeros((num_layers, n_heads))
        for layer in range(num_layers):
            for head in range(n_heads):
                kl_matrix[layer, head] = avg_kl[layer][head]
        plt.figure(figsize=(8, 6))
        plt.imshow(kl_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label="Average KL Divergence")
        plt.xlabel("Head Index")
        plt.ylabel("Decoder Layer")
        plt.title(f"Run {run_id}: Average KL Divergence per Head")
        plt.xticks(range(n_heads))
        plt.yticks(range(num_layers))
        heatmap_filename = f"{run_id}_kl_divergence_heatmap.png"
        plt.savefig(heatmap_filename)
        plt.show()
        print(f"KL divergence heatmap saved as '{heatmap_filename}'.")
    return final_circuit, kl_matrix

# ----------------------------
# Comparison Function
# ----------------------------

def compare_circuits(circuit1, circuit2):
    print("\n=== Comparison of Final Circuits ===")
    for layer in circuit1:
        heads1 = set(circuit1[layer])
        heads2 = set(circuit2.get(layer, []))
        diff = heads1.symmetric_difference(heads2)
        print(f"Layer {layer}: Difference in heads: {sorted(diff)}")

def plot_kl_difference(kl_matrix1, kl_matrix2):
    diff_matrix = np.abs(kl_matrix1 - kl_matrix2)
    plt.figure(figsize=(8, 6))
    plt.imshow(diff_matrix, cmap='magma', aspect='auto')
    plt.colorbar(label="Absolute KL Divergence Difference")
    plt.xlabel("Head Index")
    plt.ylabel("Decoder Layer")
    plt.title("KL Divergence Difference Heatmap Between Runs")
    plt.xticks(range(kl_matrix1.shape[1]))
    plt.yticks(range(kl_matrix1.shape[0]))
    plt.savefig("3_comparison_kl_difference_heatmap.png")
    plt.show()
    print("KL divergence difference heatmap saved as '3_comparison_kl_difference_heatmap.png'.")

# ----------------------------
# Main Script
# ----------------------------

def main():
    # First run: full sample uses multiples of 23, counterfactual uses multiples of 31
    final_circuit1, kl_matrix1 = run_pipeline("1", mod_main=23, mod_cf=31, n_samples=50, max_len=50)
    # Second run: full sample uses multiples of 31, counterfactual uses multiples of 23
    final_circuit2, kl_matrix2 = run_pipeline("2", mod_main=31, mod_cf=23, n_samples=50, max_len=50)
    
    compare_circuits(final_circuit1, final_circuit2)
    plot_kl_difference(kl_matrix1, kl_matrix2)
    print("\n=== Note on Circuit Patching ===")
    print("The patching functions (run_full_logits_with_patch and run_full_logits_with_circuit) correctly replace the designated heads’ context vectors with those computed from the counterfactual sample. This ensures that the influence of specific heads is suppressed (or replaced) when testing the circuit's contribution.")

if __name__ == '__main__':
    main()
