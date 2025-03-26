#!/usr/bin/env python
import argparse
import pickle
import numpy as np
import random
import torch
from modularexp.envs.arithmetic import ArithmeticEnvironment

def sample_exponentiation():
    """
    Sample a triple (a, b, c) such that:
      - a and b are chosen from modest ranges so that a**b is not astronomically large.
      - c is chosen so that a**b < c, and c remains in a reasonable range.
    """
    # Choose a and b from small ranges.
    a = random.randint(2, 10)
    b = random.randint(2, 6)
    result = a ** b
    # Choose c to be strictly greater than a**b but not too far away.
    c = random.randint(result + 1, result + 50)
    return a, b, c

def main():
    # Set up parameters.
    params = argparse.Namespace()
    params.max_len = 30            # Maximum sequence length.
    params.base = 1000             # Encoding base.
    params.max_uniform = 100       # Maximum value for uniform distribution.
    params.benford = True          # Use a logarithmic (Benford) distribution.
    params.train_uniform_exp = False  # Do not sample mod exp uniformly.
    params.train_inverse_dist = False
    params.train_sqrt_dist = False
    params.train_32_dist = False
    params.max_inverse = 100
    params.maxint = 1000000
    params.test_uniform_exp = True
    params.mixture = -1.0
    params.base = 1000
    
    env = ArithmeticEnvironment(params)
    if not hasattr(env, "rng"):
        env.rng = np.random.RandomState(42)
    
    dataset = []
    num_samples = 100  # Adjust as needed.
    for i in range(num_samples):
        a, b, c = sample_exponentiation()
        expected = a ** b

        input_tuple = (a, b, c)
        x_encoded = env.input_encoder.encode(input_tuple)
        y_encoded = env.output_encoder.encode(expected)
        
        datapoint = {
            "input": input_tuple,           # e.g., (a, b, c)
            "expected": expected,             # a**b (since a**b < c)
            "input_encoded": x_encoded,       # Tokenized/encoded input.
            "output_encoded": y_encoded,      # Tokenized/encoded expected output.
            "full_expr": input_tuple + (expected,),
        }
        dataset.append(datapoint)
    
    # Save the generated dataset to a pickle file.
    with open("activation_patching_dataset_exp.pkl", "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset generated with {len(dataset)} samples and saved to 'activation_patching_dataset_exp.pkl'.")

if __name__ == '__main__':
    main()
