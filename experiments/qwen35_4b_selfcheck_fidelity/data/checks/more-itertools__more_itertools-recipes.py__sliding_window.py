import pytest
from more_itertools import sliding_window

def test_basic_example():
    """Test the example from the docstring."""
    result = list(sliding_window(range(6), 4))
    expected = [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]
    assert result == expected

def test_shorter_than_n():
    """Test the case where iterable has fewer than n items."""
    result = list(sliding_window(range(3), 4))
    expected = []
    assert result == expected

def test_empty_iterable():
    """Test with an empty iterable."""
    result = list(sliding_window([], 2))
    expected = []
    assert result == expected

def test_n_one():
    """Test with window size of 1."""
    result = list(sliding_window(range(3), 1))
    expected = [(0,), (1,), (2,)]
    assert result == expected

def test_generator_input():
    """Test that a generator is accepted as iterable."""
    def gen():
        yield 1
        yield 2
        yield 3
    result = list(sliding_window(gen(), 2))
    expected = [(1, 2), (2, 3)]
    assert result == expected

def test_exact_n():
    """Test when iterable length equals n."""
    result = list(sliding_window(range(4), 4))
    expected = [(0, 1, 2, 3)]
    assert result == expected

def test_non_iterable():
    """Test that non-iterable input raises TypeError (implied by 'iterable' type)."""
    with pytest.raises(TypeError):
        list(sliding_window(5, 2))
