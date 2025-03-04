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
    return Dataset.from_generator(partial(generate_varying_moduli, n, max_number))


if __name__ == '__main__':
    get_dataset(1000).save_to_disk('modularexp/data')
