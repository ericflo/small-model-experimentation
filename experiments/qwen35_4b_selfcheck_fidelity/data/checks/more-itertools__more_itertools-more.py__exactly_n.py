import pytest
from more_itertools import exactly_n

def test_docstring_example_1():
    """Test the first example from the docstring."""
    assert exactly_n([True, True, False], 2) is True

def test_docstring_example_2():
    """Test the second example from the docstring."""
    assert exactly_n([True, True, False], 1) is False

def test_docstring_example_3():
    """Test the third example from the docstring."""
    assert exactly_n([0, 1, 2, 3, 4, 5], 3, lambda x: x < 3) is True

def test_default_predicate_with_integers():
    """Test default predicate=bool with integers."""
    # 0 is False, 1-5 are True. There are 5 True items.
    # n=3 should return False.
    assert exactly_n([0, 1, 2, 3, 4, 5], 3) is False

def test_empty_iterable_n_zero():
    """Test empty iterable with n=0."""
    # 0 items are True. n=0 should return True.
    assert exactly_n([], 0) is True

def test_empty_iterable_n_one():
    """Test empty iterable with n=1."""
    # 0 items are True. n=1 should return False.
    assert exactly_n([], 1) is False

def test_n_zero_with_truthy_items():
    """Test n=0 when there are truthy items."""
    # There is 1 True item. n=0 should return False.
    assert exactly_n([True, False], 0) is False

def test_n_zero_with_falsy_items():
    """Test n=0 when there are only falsy items."""
    # There are 0 True items. n=0 should return True.
    assert exactly_n([False, False], 0) is True
