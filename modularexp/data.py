from datasets import Dataset
from random import randint
from functools import partial
from typing import Generator

MAX_NUMBER = 100_000_000


# Taken from https://github.com/csknk/fast-modular-exponentiation/blob/master/python/main.py
def fast_exp(b, e, m):
    r = 1
    if 1 & e:
        r = b
    while e:
        e >>= 1
        b = (b * b) % m
        if e & 1: r = (r * b) % m
    return r


def generate_data(n: int, c: int, max_number: int = MAX_NUMBER) -> Generator[dict, None, None]:
    seen: set = set()
    i = 0
    while i < n:
        a = randint(0, max_number)
        b = randint(0, max_number)
        if (a, b) in seen:
            continue
        i += 1
        yield {"a": a, "b": b, "c": c, "y": fast_exp(a, b, c)}


def generate_varying_moduli(n: int, max_number: int = MAX_NUMBER) -> Generator[dict, None, None]:
    seen: set = set()
    i = 0
    while i < n:
        a = randint(0, max_number)
        b = randint(0, max_number)
        c = randint(0, max_number)
        if (a, b, c) in seen:
            continue
        i += 1
        yield {"a": a, "b": b, "c": c, "y": fast_exp(a, b, c)}


def get_dataset(n: int, max_number: int = MAX_NUMBER) -> Dataset:
    """Get a dataset of modular exponentiation data.

    Will generate samples of the form y = a ^ b mod c
    :param n: The amount of samples to generate
    :param max_number: The maximum number for the parameters random a,b,c
    :return: a Dataset object containing the generated data
    """
    dataset = Dataset.from_generator(partial(generate_varying_moduli, n, max_number))
    dataset = dataset.map(stringify_batch, batched=True)
    return dataset

# The below is blatantly copied from https://github.com/f-charton/Int2Int
def encode_integer(val, base=1000, digit_sep=" "):
    if val == 0:
        return '+ 0'
    sgn = '+' if val >= 0 else '-'
    val = abs(val)
    r = []
    while val > 0:
        r.append(str(val % base))
        val = val // base
    r.append(sgn)
    r.reverse()
    return digit_sep.join(r)


def encode_integer_array(x, base=1000):
    return f'V{len(x)} ' + " ".join(encode_integer(int(z), base) for z in x)


def encode_range(x):
    return str(int(x))


# end copied stuff

def stringify_batch(batch):
    batch_a,batch_b,batch_c,batch_y = batch["a"], batch["b"], batch["c"], batch["y"]
    result_prompt = []
    result_target = []
    for a,b,c,y in zip(batch_a, batch_b, batch_c, batch_y, strict=True):
        result_prompt.append(encode_integer_array([a,b,c]))
        result_target.append(encode_integer(y))
    batch["prompt"] = result_prompt
    batch["target"] = result_target
    return batch

if __name__ == '__main__':
    dataset = get_dataset(1000)
    dataset.save_to_disk('modularexp/data')
    print(dataset[0])