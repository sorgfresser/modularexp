# For a given outcome, do we always get the same prediction (no matter whether correct or wrong?)
import torch
from argparse import Namespace
import numpy as np
from modularexp.train import build_env, build_modules
from modularexp.envs.arithmetic import ArithmeticEnvironment
from modularexp.utils import to_cuda
from tqdm import tqdm

checkpoint_path = "/home/tss52/theoryofdeeplearning/modularexp/arithmetic_valid_acc_0.5.pth"

checkpoint = torch.load(checkpoint_path)
params = Namespace(**checkpoint["params"])

if isinstance(params.tasks, list):
    params.tasks = ','.join(params.tasks)

# Build the environment
env: ArithmeticEnvironment = build_env(params)
# Initialize the RNG if needed (some parts of the code expect env.rng)
if not hasattr(env, "rng"):
    seed = params.env_base_seed if hasattr(params, "env_base_seed") else 42
    env.rng = np.random.RandomState(seed)

# Build model modules
modules = build_modules(env, params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for k, v in modules.items():
    v.load_state_dict(checkpoint[k])

for key, module in modules.items():
    module.to(device)
    module.eval()

sample_count = 5_00_000
data_loader = env.create_test_iterator("test", "arithmetic", None, 512, params, size=sample_count)

predictions = torch.zeros((102, 102), dtype=torch.int64)

max_beam_length = params.max_output_len + 2
encoder = (
    modules["encoder"].module
    if params.multi_gpu
    else modules["encoder"]
)
encoder.eval()
decoder = (
    modules["decoder"].module
    if params.multi_gpu
    else modules["decoder"]
)

distinct_pairs = set()

for (x1, len1), (x2, len2), counts in tqdm(data_loader):
    # cuda
    x1_, len1_, x2, len2 = to_cuda(x1, len1, x2, len2)
    encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
    generated, _ = decoder.generate(encoded.transpose(0, 1), len1_, max_len=max_beam_length, )
    generated = generated.transpose(0, 1)

    inputs = []
    out_offset = 1
    for i in range(len(generated)):
        inputs.append(
            {
                "i": i,
                "src": x1[0: len1[i] - 1, i].tolist(),
                "tgt": x2[out_offset: len2[i] - 1, i].tolist(),
                "hyp": generated[i][out_offset:].tolist(),
            }
        )
        tgt = [env.id2word[x.item()] for x in x2[out_offset: len2[i] - 1, i]]
        tgt_value, _ = env.output_encoder.parse(tgt)
        hyp = []
        for x in generated[i][out_offset:]:
            if x == env.eos_index:
                break
            hyp.append(env.id2word[x.item()])
        hyp_value, _ = env.output_encoder.parse(hyp)
        # if tgt_value == 91:
        #     distinct_pairs.add(tuple(x1[:, i].tolist()))

        predictions[tgt_value, hyp_value] += 1


print(len(distinct_pairs))
deterministic_threshold = 0.95
deterministic = ((predictions.sum(dim=1) * deterministic_threshold) <= predictions.max(dim=1).values)
deterministic_elements = deterministic.nonzero().squeeze(1)

for value in deterministic_elements:
    print(f"{value} is deterministically predicting {predictions[value].argmax()} with {predictions[value].max()} of overall {predictions[value].sum()}")

# print(predictions[91].tolist())