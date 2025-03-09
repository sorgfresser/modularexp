# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
import numpy as np
import math
from logging import getLogger

logger = getLogger()


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


class Generator(ABC):
    def __init__(self, params):
        super().__init__()

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass


# empty for now
class Sequence(Generator):
    def __init__(self, params):
        super().__init__(params)

        self.maxint = params.maxint
        self.benford = params.benford
        self.train_uniform_exp = params.train_uniform_exp
        self.test_uniform_exp = params.test_uniform_exp
        self.max_uniform = params.max_uniform
        self.max_inverse = params.max_inverse

        self.mixture = params.mixture
        self.train_inverse_dist = params.train_inverse_dist
        self.train_sqrt_dist = params.train_sqrt_dist
        self.train_32_dist = params.train_32_dist

        self.inverse_dist = np.zeros(self.max_inverse)
        sum = 0
        if self.train_sqrt_dist:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / math.sqrt(i + 1)
                sum += 1 / math.sqrt(i + 1)
        elif self.train_32_dist:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / (i + 1) * math.sqrt(i + 1)
                sum += 1 / (i + 1) * math.sqrt(i + 1)
        else:
            for i in range(self.max_inverse):
                self.inverse_dist[i] = 1 / (i + 1)
                sum += 1 / (i + 1)
        self.inverse_dist = self.inverse_dist / sum

    def integer_sequence(self, len, rng, type=None, max=None):
        maxint = self.maxint if max is None else max
        if type == "train" and self.benford:
            # with probability 1 / maxint + 1, sample 0 instead
            p = 1 / (maxint + 1)
            zero_mask = rng.rand(len) < p
            lgs = math.log10(maxint) * rng.rand(len)
            result = np.int64(10 ** lgs)
            result[zero_mask] = 0
            return result
        return rng.randint(0, maxint + 1, len)

    def generate(self, rng: np.random.RandomState, type=None):
        mix = rng.rand() if (type == "train" and self.mixture > 0.0) else 1.0
        if type == "train" and (self.train_inverse_dist or self.train_sqrt_dist):
            out = rng.choice(range(0, self.max_inverse), p=self.inverse_dist)
            c = rng.choice(range(out + 1, self.max_inverse + 1))
            inp = self.integer_sequence(2, rng, type)
            while fast_exp(inp[0], inp[1], c) != out:
                inp = self.integer_sequence(2, rng, type, max=c)
            inp = (inp[0], inp[1], c)
        elif (type == "train" and (self.train_uniform_exp or mix < self.mixture)) or type == "test":
            out = rng.randint(0, self.max_uniform)
            c = rng.choice(range(out + 1, self.max_inverse + 1))
            inp = self.integer_sequence(2, rng, type)
            while fast_exp(inp[0], inp[1], c) != out:
                inp = self.integer_sequence(2, rng, type, max=c)
            inp = (inp[0], inp[1], c)
        else:
            c = rng.choice(range(1, self.max_inverse + 1))
            inp = self.integer_sequence(2, rng, type)
            out = fast_exp(inp[0], inp[1], c)
            inp = (inp[0], inp[1], c)
        return inp, out

    def evaluate(self, src, tgt, hyp):
        return 0, 0, 0, 0
