from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, PreTrainedTokenizer, DataCollatorWithPadding, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments
import json
from modularexp.data import get_dataset
from functools import partial
from typing import Optional, Tuple, Union
import torch

# The following doesn't work, i.e. encodes whitespaces as unknown for some reason. Need to fix maybe?
# vocab = {"<|endoftext|>": 0, "V": 1, "+": 2, "-": 3, "\t": 4}
#
# for i in range(1000):
#     vocab[str(i)] = len(vocab)
# with open("vocab.json", "w") as file:
#     json.dump(vocab, file)
#
# with open("merges.txt", "w") as file:
#     file.write("")
#
# tokenizer = GPT2Tokenizer("vocab.json", "merges.txt")
#
# print(tokenizer.encode("V3 + 61 674 854 + 87 373 34 + 5 860 374 \t + 1 733 326"))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print(tokenizer.encode("V3 + 61 674 854 + 87 373 34 + 5 860 374 \t + 1 733 326"))

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=tokenizer.model_max_length,
    n_embd=512,
    n_layer=4,
    n_head=8,
)

model = GPT2LMHeadModel(config)

data = get_dataset(1000)


def prepare_data(tokenizer: PreTrainedTokenizer, batch):
    prompt_result = tokenizer(batch["prompt"])["input_ids"]
    target_result = tokenizer(batch["target"])["input_ids"]
    sep_token = tokenizer.encode("\t")[0]

    full_result = [prompt + [sep_token] + target for prompt, target in
                   zip(prompt_result, target_result, strict=True)]

    # Mask labels, i.e. -100 for prompt, because we do not want to compute loss
    labels = [[-100] * len(prompt) + [sep_token] + target for prompt, target in
              zip(prompt_result, target_result, strict=True)]

    lengths = [len(x) for x in full_result]
    # max_len = max(lengths)

    # attention_mask = torch.tensor([[1] * length + [0] * (max_len - length) for length in lengths], dtype=torch.long)
    # input_ids = torch.tensor([full + [tokenizer.pad_token_id] * (max_len - len(full)) for full in full_result], dtype=torch.long)

    attention_mask = [[1] * length for length in lengths]
    input_ids = full_result

    batch["attention_mask"] = attention_mask
    batch["input_ids"] = input_ids
    batch["labels"] = target_result
    return batch


data = data.map(partial(prepare_data, tokenizer), batched=True)
split_data = data.train_test_split(0.2)

args = Seq2SeqTrainingArguments(output_dir="output",
                                report_to=["wandb"], num_train_epochs=10)

trainer = Seq2SeqTrainer(model, args, DataCollatorWithPadding(tokenizer), split_data["train"], split_data["test"])

trainer.train()
