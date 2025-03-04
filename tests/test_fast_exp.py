from random import randint
from modularexp.data import MAX_NUMBER, fast_exp

def compute_mod_slow(a, b, c):
    result = 1
    for _ in range(b):
        result = (result * a) % c
    return result


def test_small():
    assert fast_exp(10, 10, 3) == compute_mod_slow(10, 10, 3)

def test_zero():
    assert fast_exp(0, 10, 4) == compute_mod_slow(0, 10, 4)
    assert fast_exp(10, 0, 4) == compute_mod_slow(10, 0, 4)
    assert fast_exp(0, 0, 4) == compute_mod_slow(0, 0, 4)


def test_large_scale_automatically():
    for _ in range(100):
        a = randint(0, MAX_NUMBER / 10)
        b = randint(0, MAX_NUMBER / 10)
        c = randint( 0, MAX_NUMBER / 10)

        assert fast_exp(a, b, c) == compute_mod_slow(a, b, c)
