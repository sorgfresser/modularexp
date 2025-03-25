import os
import math
import random
import re
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_ckpt_value(ckpt, key):
    """
    Retrieve the value for 'key' from ckpt.
    If not found, toggle the "transformer." prefix.
    """
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
    """
    Transpose the weight tensor if its shape matches the transposed expected_shape.
    """
    if weight.shape == expected_shape:
        return weight
    elif weight.shape == (expected_shape[1], expected_shape[0]):
        return weight.transpose(0, 1)
    else:
        print(f"Warning: unexpected shape {weight.shape} (expected {expected_shape}).")
        return weight

def remap_checkpoint(ckpt, config):
    """
    Remap checkpoint keys from the custom transformer (without a "transformer." prefix)
    to the keys expected by GPT2LMHeadModel. Transposes weights for linear layers as needed.
    """
    new_state = {}
    # Remap embedding weights.
    new_state["transformer.wte.weight"] = get_ckpt_value(ckpt, "embeddings.weight")
    new_state["transformer.wpe.weight"] = get_ckpt_value(ckpt, "position_embeddings.weight")

    n_layers = config.n_layer  # use number of layers from config
    for i in range(n_layers):
        # Use "layers.{i}" from the checkpoint instead of "transformer.layers.{i}"
        ckpt_prefix = f"layers.{i}"
        gpt2_prefix = f"transformer.h.{i}"

        # Layer norm 1.
        new_state[f"{gpt2_prefix}.ln_1.weight"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.layer_norm1.weight")
        new_state[f"{gpt2_prefix}.ln_1.bias"]   = get_ckpt_value(ckpt, f"{ckpt_prefix}.layer_norm1.bias")

        # Concatenate query, key, and value weights.
        q_w = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.q_lin.weight")
        k_w = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.k_lin.weight")
        v_w = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.v_lin.weight")
        cat_w = torch.cat([q_w, k_w, v_w], dim=0)
        new_state[f"{gpt2_prefix}.attn.c_attn.weight"] = maybe_transpose(cat_w, (config.n_embd, 3 * config.n_embd))

        q_b = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.q_lin.bias")
        k_b = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.k_lin.bias")
        v_b = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.v_lin.bias")
        new_state[f"{gpt2_prefix}.attn.c_attn.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        # Self-attention output.
        w_proj = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.out_lin.weight")
        new_state[f"{gpt2_prefix}.attn.c_proj.weight"] = maybe_transpose(w_proj, (config.n_embd, config.n_embd))
        new_state[f"{gpt2_prefix}.attn.c_proj.bias"]   = get_ckpt_value(ckpt, f"{ckpt_prefix}.self_attention.out_lin.bias")

        # Layer norm 2.
        new_state[f"{gpt2_prefix}.ln_2.weight"] = get_ckpt_value(ckpt, f"{ckpt_prefix}.layer_norm2.weight")
        new_state[f"{gpt2_prefix}.ln_2.bias"]   = get_ckpt_value(ckpt, f"{ckpt_prefix}.layer_norm2.bias")

        # MLP layers.
        w_fc = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.lin1.weight")
        new_state[f"{gpt2_prefix}.mlp.c_fc.weight"] = maybe_transpose(w_fc, (config.n_embd, 4 * config.n_embd))
        new_state[f"{gpt2_prefix}.mlp.c_fc.bias"]   = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.lin1.bias")

        w_proj_mlp = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.lin2.weight")
        new_state[f"{gpt2_prefix}.mlp.c_proj.weight"] = maybe_transpose(w_proj_mlp, (4 * config.n_embd, config.n_embd))
        new_state[f"{gpt2_prefix}.mlp.c_proj.bias"]   = get_ckpt_value(ckpt, f"{ckpt_prefix}.ffn.lin2.bias")

    # Final layer norm (if present in checkpoint)
    try:
        new_state["transformer.ln_f.weight"] = get_ckpt_value(ckpt, "layer_norm_emb.weight")
        new_state["transformer.ln_f.bias"]   = get_ckpt_value(ckpt, "layer_norm_emb.bias")
    except KeyError:
        new_state["transformer.ln_f.weight"] = torch.ones(new_state["transformer.wte.weight"].size(1))
        new_state["transformer.ln_f.bias"]   = torch.zeros(new_state["transformer.wte.weight"].size(1))
        print("Warning: layer_norm_emb not found; using default values.")

    # Tie lm_head.weight to transformer.wte.weight.
    new_state["lm_head.weight"] = new_state["transformer.wte.weight"]

    return new_state

def load_model(checkpoint_path):
    """
    Loads the GPT2LMHeadModel with a configuration matching the checkpoint,
    remaps checkpoint keys (if needed), and sets the model to output hidden states.
    """
    if os.path.exists("vocab.json") and os.path.exists("merges.txt"):
        tokenizer = GPT2Tokenizer("vocab.json", "merges.txt")
    else:
        # Dummy tokenizer fallback.
        class DummyTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
                self.eos_token = "<|endoftext|>"
            def encode(self, text, return_tensors=None):
                tokens = [int(t) if t.isdigit() else ord(t) for t in text.split()]
                if return_tensors == "pt":
                    return torch.tensor([tokens])
                return tokens
            def decode(self, ids):
                return " ".join(str(i) for i in ids)
        tokenizer = DummyTokenizer(1020)

    if hasattr(tokenizer, "pad_token"):
        tokenizer.pad_token = tokenizer.eos_token if hasattr(tokenizer, "eos_token") else "<|endoftext|>"

    # Create configuration and enable output_hidden_states.
    config = GPT2Config(
        vocab_size=1020,
        n_positions=4096,
        n_embd=256,
        n_layer=4,
        n_head=8,
        output_hidden_states=True  # Needed for decoder visualization.
    )
    model = GPT2LMHeadModel(config)

    # Load checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    
    # Check for the "encoder" key and use it for remapping.
    if "encoder" in checkpoint:
        print("Found 'encoder' in checkpoint. Remapping keys...")
        encoder_ckpt = checkpoint["encoder"]
        new_state_dict = remap_checkpoint(encoder_ckpt, config)
        model.load_state_dict(new_state_dict)
    else:
        print("No 'encoder' key found; attempting to load full checkpoint directly.")
        model.load_state_dict(checkpoint)

    model.eval()
    return model, tokenizer


def visualize_numeric_embeddings(
    model,
    tokenizer,
    color_scheme="value",
    max_token=None,
    label_stride=1,
    n_components=2
):
    """
    Perform PCA on the model's token embedding matrix and plot the 2D (or nD) projection 
    for numeric tokens. By default, we consider numeric tokens in the range [1..max_token],
    but if max_token is None, we include *all* positive numeric tokens. Points can be 
    colored by various numeric properties.
    
    The available color schemes are:
      - "value": Colors by the numeric token's value.
      - "prime": Colors by whether the number is prime (1 for prime, 0 otherwise).
      - "parity": Colors by parity (0 for even, 1 for odd).
      - "lowest_prime_factor": Colors by the smallest prime factor.
      - "divisor_count": Colors by the number of divisors.
      - "totient": Colors by Euler's totient function φ(n).
      - "multiplicative_order": Colors by the multiplicative order modulo a fixed small modulus (default mod 7).
      - "primitive_root": Colors tokens as 1 if they are primitive roots modulo a fixed prime (default prime 11), 0 otherwise.
      - "residue": Colors by the residue modulo a fixed small integer (default mod 5).

    Args:
        model: The GPT-2 model (with embeddings in model.transformer.wte.weight).
        tokenizer: The associated tokenizer to decode token IDs.
        color_scheme (str): How to color numeric tokens. See above.
        max_token (int or None): If given, only show tokens <= max_token. If None, show all.
        label_stride (int): Annotate every nth numeric token to reduce clutter (1 = label all).
        n_components (int): Number of PCA components. Typically 2 for a scatter plot.
    """
    # -------------------------- HELPER FUNCTIONS --------------------------
    def is_prime(n):
        """Return True if n is a prime (simple check). Assumes n >= 2."""
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        r = int(n**0.5)
        for i in range(5, r+1, 6):
            if n % i == 0 or n % (i + 2) == 0:
                return False
        return True

    def get_lowest_prime_factor(n):
        """
        Return the smallest prime factor of n.
        If n <= 1, or if no factor is found, return 1 as a fallback.
        """
        if n < 2:
            return 1
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        r = int(n**0.5)
        for i in range(5, r+1, 6):
            if n % i == 0:
                return i
            if n % (i+2) == 0:
                return i+2
        return n

    def divisor_count(n):
        """Return the number of divisors of n."""
        count = 0
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                count += 1 if i * i == n else 2
        return count

    def totient(n):
        """Return Euler's totient function φ(n)."""
        result = n
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result

    def multiplicative_order(a, m):
        """
        Return the multiplicative order of a modulo m (the smallest exponent k with a^k ≡ 1 mod m).
        If a and m are not coprime, return 0.
        """
        if math.gcd(a, m) != 1:
            return 0
        order = 1
        current = a % m
        while current != 1:
            current = (current * a) % m
            order += 1
            if order > m:  # fail safe
                return 0
        return order

    embeddings = model.transformer.wte.weight.detach().cpu().numpy()
    token_count = embeddings.shape[0]
    pca = PCA(n_components=n_components)
    embeddings_proj = pca.fit_transform(embeddings)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    tokens = [tokenizer.decode([i]).strip() for i in range(token_count)]
    numeric_indices = []
    numeric_values = []
    for i, token in enumerate(tokens):
        if token.isdigit():
            val = int(token)
            if max_token is not None:
                if 1 <= val <= max_token:
                    numeric_indices.append(i)
                    numeric_values.append(val)
            else:
                if val > 0:
                    numeric_indices.append(i)
                    numeric_values.append(val)
    numeric_points = embeddings_proj[numeric_indices]
    numeric_values = np.array(numeric_values)
    if color_scheme == "value":
        color_vals = numeric_values
        color_label = "Numeric Token Value"
    elif color_scheme == "prime":
        color_vals = np.array([1 if is_prime(v) else 0 for v in numeric_values])
        color_label = "Is Prime? (prime=1, composite=0)"
    elif color_scheme == "parity":
        color_vals = numeric_values % 2
        color_label = "Parity (0=even, 1=odd)"
    elif color_scheme == "lowest_prime_factor":
        color_vals = np.array([get_lowest_prime_factor(v) for v in numeric_values])
        color_label = "Lowest Prime Factor"
    elif color_scheme == "divisor_count":
        color_vals = np.array([divisor_count(v) for v in numeric_values])
        color_label = "Number of Divisors"
    elif color_scheme == "totient":
        color_vals = np.array([totient(v) for v in numeric_values])
        color_label = "Euler's Totient φ(n)"
    elif color_scheme == "multiplicative_order":
        mod = 7
        color_vals = np.array([multiplicative_order(v, mod) for v in numeric_values])
        color_label = f"Multiplicative Order mod {mod}"
    elif color_scheme == "primitive_root":
        p = 11
        color_vals = np.array([1 if (v % p != 0 and multiplicative_order(v, p) == p-1) else 0 for v in numeric_values])
        color_label = f"Primitive Root mod {p} (1 if yes, 0 if no)"
    elif color_scheme == "residue":
        mod = 5
        color_vals = numeric_values % mod
        color_label = f"Residue mod {mod}"
    else:
        raise ValueError(f"Unknown color_scheme '{color_scheme}'.")
    if n_components == 2:
        plt.figure(figsize=(12, 8))
        sc = plt.scatter(
            numeric_points[:, 0], 
            numeric_points[:, 1],
            c=color_vals, 
            s=50, 
            alpha=0.8, 
            cmap='viridis'
        )
        plt.title(f"PCA of Numeric Token Embeddings (2D, color by {color_scheme})")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        for idx, val in enumerate(numeric_values):
            if idx % label_stride == 0:
                x, y = numeric_points[idx, 0], numeric_points[idx, 1]
                plt.annotate(str(val), (x, y), fontsize=9, xytext=(5, 2), textcoords="offset points")
        plt.grid(True)
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            numeric_points[:, 0], 
            numeric_points[:, 1],
            numeric_points[:, 2],
            c=color_vals,
            s=50,
            alpha=0.8,
            cmap='viridis'
        )
        ax.set_title(f"PCA of Numeric Token Embeddings (3D, color by {color_scheme})")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        for idx, val in enumerate(numeric_values):
            if idx % label_stride == 0:
                x, y, z = numeric_points[idx, 0], numeric_points[idx, 1], numeric_points[idx, 2]
                ax.text(x, y, z, str(val), fontsize=8)
    cb = plt.colorbar(sc)
    cb.set_label(color_label)
    plt.show()

def visualize_decoder_hidden_states(model, tokenizer, prompt, layer_idx=-1):
    """
    Visualizes the hidden states (decoder outputs) from a specified layer.
    Reduces hidden state dimensionality via PCA and annotates each token.
    
    Parameters:
      prompt (str): The input text to feed into the model.
      layer_idx (int): Which layer's hidden states to visualize (default: last layer).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    selected_hidden = hidden_states[layer_idx].squeeze(0).detach().cpu().numpy()
    pca = PCA(n_components=2)
    hidden_2d = pca.fit_transform(selected_hidden)
    
    tokens = [tokenizer.decode([t]).strip() for t in input_ids[0].tolist()]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(hidden_2d[:, 0], hidden_2d[:, 1], s=50, alpha=0.7, label="Token Representations")
    for i, token in enumerate(tokens):
        plt.annotate(token, (hidden_2d[i, 0], hidden_2d[i, 1]), fontsize=10)
    
    plt.title(f"PCA of Decoder Outputs from Layer {layer_idx if layer_idx >= 0 else 'last'}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    checkpoint_path = r"modularexp\checkpoints\checkpoint.pth"
    model, tokenizer = load_model(checkpoint_path)
    components = 3

    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="value",
        max_token=None,
        label_stride=1,
        n_components=components
    )
    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="prime",
        max_token=100,
        label_stride=1,
        n_components=components
    )
    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="parity",
        max_token=None,
        n_components=components
    )
    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="lowest_prime_factor",
        max_token=200,
        n_components=components
    )
    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="divisor_count",
        max_token=200,
        label_stride=1,
        n_components=components
    )
    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="totient",
        max_token=200,
        label_stride=1,
        n_components=components
    )
    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="multiplicative_order",
        max_token=200,
        label_stride=1,
        n_components=components
    )
    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="primitive_root",
        max_token=200,
        label_stride=1,
        n_components=components
    )
    visualize_numeric_embeddings(
        model, 
        tokenizer,
        color_scheme="residue",
        max_token=200,
        label_stride=1,
        n_components=components
    )

if __name__ == "__main__":
    main()