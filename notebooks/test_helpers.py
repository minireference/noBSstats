"""
Tests for the function exercises in `python_tutorial.ipynb`.
"""
import math
import random
import statistics


def random_list(n=10, min=0.0, max=100.0):
    """
    Returns a list of length `n` of random floats between `min` and `max`.
    """
    values = []
    for i in range(n):
        r = random.random()
        value = min + r*(max-min)
        values.append(value)
    return values


def test_mean(function):
    """
    Run a few lists to check if value returned by `function` matches expected.
    """
    assert function([1,1,1]) == 1
    assert function([61,72,85,92]) == 77.5
    list10 = random_list(n=10)
    assert math.isclose(function(list10), statistics.mean(list10))
    list100 = random_list(n=100)
    assert math.isclose(function(list100), statistics.mean(list100))
    print("All tests passed. Good job!")



def test_std(function):
    """
    Run a few lists to check if value returned by `function` matches expected.
    """
    import math, statistics
    assert function([1,1,1]) == 0
    assert math.isclose(function([61,72,85,92]), 13.771952173409064)
    list10 = random_list(n=10)
    assert math.isclose(function(list10), statistics.stdev(list10))
    list100 = random_list(n=100)
    assert math.isclose(function(list100), statistics.stdev(list100))
    print("All tests passed. Good job!")
