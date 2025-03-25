#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse

from modularexp.model.transformer import TransformerModel
from modularexp.envs.arithmetic import ArithmeticEnvironment
from modularexp.utils import bool_flag  # assumes bool_flag is defined

#########################################
# Checkpoint Remapping Functions
#########################################

def get_ckpt_value(ckpt, key):
    """Retrieve the value for 'key' from ckpt, trying an alternate prefix if needed."""
    if key in ckpt:
        return ckpt[key]
    if key.startswith("transformer."):
        alt_key = key[len("transformer."):]
    else:
        alt_key = "transformer." + key
    if alt_key in ckpt:
        return ckpt[alt_key]
    raise KeyError(f"Key {key} not found in checkpoint; also tried {alt_key}.")

def maybe_transpose(weight, expected_shape):
    """Transpose weight if necessary."""
    if weight.shape == expected_shape:
        return weight
    elif weight.shape == (expected_shape[1], expected_shape[0]):
        return weight.transpose(0, 1)
    else:
        print(f"Warning: unexpected shape {weight.shape} (expected {expected_shape}).")
        return weight

def remap_checkpoint(ckpt, config):
    """
    Remap keys from a training checkpoint to match the model's expected key names.
    Splits merged self-attention weights if needed.
    """
    new_state = {}
    new_state["position_embeddings.weight"] = get_ckpt_value(ckpt, "transformer.position_embeddings.weight")
    new_state["embeddings.weight"] = get_ckpt_value(ckpt, "transformer.embeddings.weight")
    
    n_layers = config.n_enc_layers if hasattr(config, "n_enc_layers") else config.n_layers
    for i in range(n_layers):
        ckpt_prefix = f"transformer.layers.{i}"
        model_prefix = f"layers.{i}"
        new_state[f"{model_prefix}.layer_norm1.weight"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.layer_norm1.weight")
        new_state[f"{model_prefix}.layer_norm1.bias"]   = get_ckpt_value(ckpt, f"{ckpt_prefix}.layer_norm1.bias")
        merged_w = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.q_lin.weight")
        if merged_w.shape[1] == config.enc_emb_dim * 3:
            q_w, k_w, v_w = torch.split(merged_w, config.enc_emb_dim, dim=1)
        else:
            q_w = merged_w
            k_w = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.k_lin.weight")
            v_w = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.v_lin.weight")
        new_state[f"{model_prefix}.self_attention.q_lin.weight"] = maybe_transpose(q_w, (config.enc_emb_dim, config.enc_emb_dim))
        new_state[f"{model_prefix}.self_attention.k_lin.weight"] = maybe_transpose(k_w, (config.enc_emb_dim, config.enc_emb_dim))
        new_state[f"{model_prefix}.self_attention.v_lin.weight"] = maybe_transpose(v_w, (config.enc_emb_dim, config.enc_emb_dim))
        merged_b = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.q_lin.bias")
        if merged_b.shape[0] == config.enc_emb_dim * 3:
            q_b, k_b, v_b = torch.split(merged_b, config.enc_emb_dim, dim=0)
        else:
            q_b = merged_b
            k_b = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.k_lin.bias")
            v_b = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.v_lin.bias")
        new_state[f"{model_prefix}.self_attention.q_lin.bias"] = q_b
        new_state[f"{model_prefix}.self_attention.k_lin.bias"] = k_b
        new_state[f"{model_prefix}.self_attention.v_lin.bias"] = v_b
        w_proj = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.out_lin.weight")
        new_state[f"{model_prefix}.self_attention.out_lin.weight"] = maybe_transpose(w_proj, (config.enc_emb_dim, config.enc_emb_dim))
        new_state[f"{model_prefix}.self_attention.out_lin.bias"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.out_lin.bias")
        new_state[f"{model_prefix}.layer_norm2.weight"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.layer_norm2.weight")
        new_state[f"{model_prefix}.layer_norm2.bias"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.layer_norm2.bias")
        w_fc = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.lin1.weight")
        new_state[f"{model_prefix}.ffn.lin1.weight"] = maybe_transpose(w_fc, (config.enc_emb_dim * 4, config.enc_emb_dim))
        new_state[f"{model_prefix}.ffn.lin1.bias"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.lin1.bias")
        midlin_key = f"{ckpt_prefix}.ffn.midlin.0.weight"
        if midlin_key in ckpt:
            new_state[f"{model_prefix}.ffn.midlin.0.weight"] = get_ckpt_value(ckpt, midlin_key)
            new_state[f"{model_prefix}.ffn.midlin.0.bias"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.midlin.0.bias")
        else:
            new_state[f"{model_prefix}.ffn.midlin.0.weight"] = torch.zeros(config.enc_emb_dim * 4, config.enc_emb_dim * 4)
            new_state[f"{model_prefix}.ffn.midlin.0.bias"] = torch.zeros(config.enc_emb_dim * 4)
        w_proj_mlp = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.lin2.weight")
        new_state[f"{model_prefix}.ffn.lin2.weight"] = maybe_transpose(w_proj_mlp, (config.enc_emb_dim, config.enc_emb_dim * 4))
        new_state[f"{model_prefix}.ffn.lin2.bias"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.lin2.bias")
    
    try:
        new_state["layer_norm_emb.weight"] = get_ckpt_value(ckpt, "transformer.layer_norm_emb.weight")
        new_state["layer_norm_emb.bias"] = get_ckpt_value(ckpt, "transformer.layer_norm_emb.bias")
    except KeyError:
        emb_weight = new_state["embeddings.weight"]
        new_state["layer_norm_emb.weight"] = torch.ones(emb_weight.size(1))
        new_state["layer_norm_emb.bias"] = torch.zeros(emb_weight.size(1))
        print("Warning: transformer.layer_norm_emb not found; using defaults.")
    
    new_state["proj.weight"] = new_state["embeddings.weight"]
    if "transformer.proj.bias" in ckpt:
        new_state["proj.bias"] = get_ckpt_value(ckpt, "transformer.proj.bias")
    else:
        out_dim = new_state["embeddings.weight"].shape[0]
        new_state["proj.bias"] = torch.zeros(out_dim)
    
    return new_state

#########################################
# Model Loading
#########################################

def load_model(checkpoint_path, params, id2word):
    """Load the TransformerModel from a checkpoint, remapping keys if needed."""
    model = TransformerModel(params, id2word, is_encoder=True, with_output=True)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "encoder" in checkpoint:
        print("Found 'encoder' in checkpoint. Remapping keys...")
        encoder_ckpt = checkpoint["encoder"]
        new_state_dict = remap_checkpoint(encoder_ckpt, params)
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

#########################################
# Custom Dataset for Activation Patching
#########################################

class PatchingDataset(Dataset):
    """
    Dataset for activation patching using the ArithmeticEnvironment.
    Each sample is a tuple: (input sequence, output sequence, extra info).
    (Extra info is expected to contain the numbers a, b, c, and d for modular exponentiation.)
    """
    def __init__(self, env, task, train, params, path=None, size=None, data_type=None):
        super(PatchingDataset, self).__init__()
        self.env = env
        self.task = task
        self.train = train
        self.size = 10000
        self.data_type = data_type  # "train", "valid", or "test"

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = self.env.gen_expr(self.data_type, self.task)
        while sample is None:
            sample = self.env.gen_expr(self.data_type, self.task)
        return sample  # (x, y, info)

    def collate_fn(self, elements):
        x, y, _ = zip(*elements)
        nb_eqs = [self.env.code_class(xi, yi) for xi, yi in zip(x, y)]
        x_ids = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
        y_ids = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
        x, x_len = self.batch_sequences(x_ids, self.env.pad_index, self.env.eos_index, self.env.eos_index, no_bos=True)
        y, y_len = self.batch_sequences(y_ids, self.env.pad_index, self.env.eos_index, self.env.eos_index, no_bos=False)
        dummy_gate = torch.zeros(len(elements), dtype=torch.long)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_eqs), dummy_gate

    def batch_sequences(self, sequences, pad_index, bos_index, eos_index, no_bos=False):
        initial_offset = 0 if no_bos else 1
        lengths = torch.LongTensor([len(s) + 1 + initial_offset for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(pad_index)
        if not no_bos:
            sent[0] = bos_index
        for i, s in enumerate(sequences):
            sent[initial_offset:lengths[i]-1, i].copy_(s)
            sent[lengths[i]-1, i] = eos_index
        return sent, lengths

#########################################
# Activation Collection & Patching Evaluation
#########################################

def activation_collection_and_patching_evaluation(model, dataloader, save_dir="modexp/activations"):
    """
    Runs a forward pass to collect attention head activations from each transformer layer,
    performs injection patching on a target layer, and saves the collected activations and logits.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    baseline_activations = {i: [] for i in range(len(model.layers))}
    patched_activations = {i: [] for i in range(len(model.layers))}
    
    def make_hook(storage_dict, layer_idx):
        def hook(module, input, output):
            bs, qlen, dim = output.shape
            n_heads = module.n_heads
            dim_per_head = dim // n_heads
            head_act = output.view(bs, qlen, n_heads, dim_per_head).permute(0, 2, 1, 3)
            storage_dict[layer_idx].append(head_act.detach().cpu())
        return hook

    # Baseline forward pass (using input tokens)
    baseline_hooks = []
    for idx, layer in enumerate(model.layers):
        h = layer.self_attention.register_forward_hook(make_hook(baseline_activations, idx))
        baseline_hooks.append(h)
    
    for batch in dataloader:
        (x, x_len), (y, y_len), nb_eqs, _ = batch
        print(f"Baseline input sequence lengths: {x_len.tolist()}")
        baseline_hidden = model.fwd(x=x, lengths=x_len, causal=True,
                                    src_enc=None, src_len=None, positions=None,
                                    use_cache=False)
        baseline_logits = model.proj(baseline_hidden)
        break
    for h in baseline_hooks:
        h.remove()
    
    # Patched forward pass (using target tokens)
    patched_hooks = []
    for idx, layer in enumerate(model.layers):
        h = layer.self_attention.register_forward_hook(make_hook(patched_activations, idx))
        patched_hooks.append(h)
    
    print(f"Patched input sequence lengths: {y_len.tolist()}")
    patched_hidden = model.fwd(x=y, lengths=y_len, causal=True,
                               src_enc=None, src_len=None, positions=None,
                               use_cache=False)
    patched_logits = model.proj(patched_hidden)
    for h in patched_hooks:
        h.remove()
    
    # Injection patching on a target layer
    target_layer_index = 2  # adjust as needed
    target_module = model.layers[target_layer_index].self_attention
    if patched_activations[target_layer_index]:
        patched_act = patched_activations[target_layer_index][0]  # shape: (batch, n_heads, seq_len, dim_per_head)
    else:
        raise RuntimeError("No patched activation collected for target layer.")
    patched_act_unshaped = patched_act.transpose(1, 2).contiguous().view(patched_act.size(0), patched_act.size(2), -1)
    min_len = y_len[0].item()  # e.g. 4 tokens
    x_injection = x[:min_len, :]
    injection_lengths = torch.full((x_injection.shape[1],), min_len, dtype=x_len.dtype, device=x_len.device)
    
    original_forward = target_module.forward
    target_module.forward = lambda *args, **kwargs: patched_act_unshaped
    injected_hidden = model.fwd(x=x_injection, lengths=injection_lengths, causal=True,
                                src_enc=None, src_len=None, positions=None,
                                use_cache=False)
    injected_logits = model.proj(injected_hidden)
    target_module.forward = original_forward
    
    activations_dict = {
        "baseline_activations": {k: torch.cat(v, dim=0) if v else None for k, v in baseline_activations.items()},
        "patched_activations": {k: torch.cat(v, dim=0) if v else None for k, v in patched_activations.items()},
        "baseline_logits": baseline_logits.detach().cpu(),
        "patched_logits": patched_logits.detach().cpu(),
        "injected_logits": injected_logits.detach().cpu(),
    }
    save_path = os.path.join(save_dir, "activations.pth")
    torch.save(activations_dict, save_path)
    print(f"Saved activations and logits to {save_path}")
    
    print("Baseline activation shapes:")
    for layer_idx, acts in baseline_activations.items():
        if acts:
            print(f"Layer {layer_idx}: {acts[0].shape}")
    print("Patched activation shapes:")
    for layer_idx, acts in patched_activations.items():
        if acts:
            print(f"Layer {layer_idx}: {acts[0].shape}")
    
    print("Baseline logits (first 5 values of token 0):", baseline_logits[0, :5])
    print("Patched logits (first 5 values of token 0):", patched_logits[0, :5])
    print("Injected logits (first 5 values of token 0):", injected_logits[0, :5])

#########################################
# Sweep Experiment for Modular Exponentiation
#########################################

def gen_expr_with_params(env, data_type, task, param_dict):
    """
    Helper to generate a modular exponentiation sample with given parameters.
    Since env.gen_expr() does not support extra keyword arguments, we call it normally
    and then override the info field with our chosen parameters.
    Also, we clip 'a' and 'b' to be within [0, 10^6).
    """
    # Enforce bounds on a and b
    param_dict["a"] = max(0, min(int(param_dict["a"]), 10**6 - 1))
    param_dict["b"] = max(0, min(int(param_dict["b"]), 10**6 - 1))
    # Generate a sample normally
    sample = env.gen_expr(data_type, task)
    if sample is None:
        return None
    x, y, _ = sample
    # Override info with our chosen parameters (as a tuple)
    info = (param_dict["a"], param_dict["b"], param_dict["c"], param_dict["d"])
    return (x, y, info)

def sweep_experiment(model, env, base_sample, data_type="train", task="arithmetic"):
    """
    For a given base sample (assumed to be (x, y, info) with x, y as lists of tokens
    and info as a tuple/list with at least 4 elements: (a, b, c, d)),
    vary each parameter by ±1 and compute a metric (logit difference for the correct answer token).
    """
    x_base, y_base, info = base_sample
    if isinstance(info, (list, tuple)) and len(info) >= 4:
        a, b, c, d = info[:4]
        base_params = {"a": a, "b": b, "c": c, "d": d}
    else:
        raise ValueError("Expected info to be a tuple/list with at least 4 elements (a, b, c, d).")
    
    # Convert x_base (list of tokens) to a tensor with a singleton batch dimension.
    x_tensor = torch.LongTensor([env.word2id[w] for w in x_base]).unsqueeze(1)  # shape (slen, 1)
    x_len = torch.LongTensor([x_tensor.size(0)])
    
    # Run a baseline forward pass on the base sample
    base_hidden = model.fwd(x=x_tensor, lengths=x_len, causal=True,
                            src_enc=None, src_len=None, positions=None,
                            use_cache=False)
    base_logits = model.proj(base_hidden)
    
    print("Base parameters:", base_params)
    results = {}
    for param in ["a", "b", "c", "d"]:
        results[param] = {}
        for delta in [-1, 1]:
            mod_params = base_params.copy()
            mod_params[param] += delta
            print(f"Generating sample with {param} changed by {delta}: {mod_params}")
            mod_sample = gen_expr_with_params(env, data_type, task, mod_params)
            if mod_sample is None:
                print(f"Could not generate sample for {mod_params}")
                continue
            x_mod, y_mod, info_mod = mod_sample
            x_mod_tensor = torch.LongTensor([env.word2id[w] for w in x_mod]).unsqueeze(1)
            x_mod_len = torch.LongTensor([x_mod_tensor.size(0)])
            mod_hidden = model.fwd(x=x_mod_tensor, lengths=x_mod_len, causal=True,
                                   src_enc=None, src_len=None, positions=None,
                                   use_cache=False)
            mod_logits = model.proj(mod_hidden)
            # Assume the correct token is given by the fourth element of info_mod (vocabulary index)
            if isinstance(info_mod, (list, tuple)) and len(info_mod) >= 4:
                correct_token = info_mod[3]
            else:
                raise ValueError("Expected info_mod to have at least 4 elements (a, b, c, d).")
            # Index the final token in the sequence (dimension 1) and then the vocabulary dimension (dimension 2)
            base_logit = base_logits[0, -1, correct_token].item()
            mod_logit = mod_logits[0, -1, correct_token].item()
            results[param][delta] = mod_logit - base_logit
            print(f"Param {param} delta {delta}: logit difference = {mod_logit - base_logit:.3f}")
    sweep_save_path = os.path.join("modexp/activations", "sweep_results.pth")
    torch.save(results, sweep_save_path)
    print(f"Sweep experiment results saved to {sweep_save_path}")
    return results


#########################################
# Main Execution
#########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_path", type=str, default="localpath", help="Experiment dump path")
    parser.add_argument("--cpu", type=bool_flag, default=False, help="Run on CPU")
    parser.add_argument("--wandb", type=str, default="", help="Wandb API key")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--batch_size_eval", type=int, default=1024, help="Batch size for evaluation")
    parser.add_argument("--wandb_run", type=str, default="", help="Wandb run ID")
    parser.add_argument("--reload_checkpoint", type=str, default="modularexp/checkpoints/checkpoint.pth", help="Path to the checkpoint to reload")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    ArithmeticEnvironment.register_args(parser)
    args = parser.parse_args()
    
    class DummyParams:
        max_len = args.max_len
        base = args.base
        max_uniform = args.max_uniform
        maxint = args.maxint
        benford = args.benford
        train_uniform_exp = args.train_uniform_exp
        train_inverse_dist = args.train_inverse_dist
        train_sqrt_dist = args.train_sqrt_dist
        train_32_dist = args.train_32_dist
        max_inverse = args.max_inverse
        test_uniform_exp = args.test_uniform_exp
        mixture = args.mixture
        n_words = None  
        eos_index = None
        pad_index = None
        sep_index = None
        enc_emb_dim = 256
        dec_emb_dim = 256
        n_enc_hidden_layers = 2
        n_dec_hidden_layers = 2
        n_enc_heads = 8
        n_dec_heads = 8
        n_enc_layers = 4
        n_dec_layers = 4
        dropout = 0.1
        attention_dropout = 0.1
        norm_attention = False
        sinusoidal_embeddings = False
        xav_init = False
        share_inout_emb = True
        fp16 = False
        batch_size = args.batch_size
        batch_size_eval = args.batch_size_eval
        dump_path = args.dump_path
        wandb = args.wandb
        wandb_run = args.wandb_run
        reload_checkpoint = args.reload_checkpoint

    params = DummyParams()
    checkpoint_path = params.reload_checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    env = ArithmeticEnvironment(params)
    env.rng = np.random.RandomState(42)
    id2word = env.id2word
    model = load_model(checkpoint_path, params, id2word)
    print("Model loaded and set to eval mode.")
    
    dataset = PatchingDataset(env=env, task="arithmetic", train=True, params=params, path=None, data_type="train")
    dataloader = DataLoader(dataset, batch_size=params.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
    
    # Run activation collection and patching evaluation
    activation_collection_and_patching_evaluation(model, dataloader)
    
    # Run sweep experiment on one sample to vary a, b, c, and d
    sample = dataset[0]
    print("Running sweep experiment on a sample...")
    sweep_results = sweep_experiment(model, env, sample)
    print("Sweep experiment results:", sweep_results)
    
if __name__ == "__main__":
    main()
