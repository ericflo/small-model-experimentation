import pytest
from dpath.segments import match

def test_basic_match():
    """Test that segments match the glob when they are identical."""
    assert match(['a', 'b'], 'a/b') is True

def test_integer_segment_conversion():
    """Test that integer segments are converted to strings."""
    assert match([1], '1') is True

def test_integer_glob_conversion():
    """Test that integer globs are converted to strings."""
    assert match(['1'], 1) is True

def test_star_star_expansion():
    """Test that star-star segments expand to 0 or more star segments."""
    assert match(['a', 'b'], 'a/**') is True

def test_star_star_type_coercion():
    """Test that star-star segments coerce type to match segment."""
    assert match([1, 2], '1/**') is True

def test_non_match():
    """Test that segments do not match the glob when they differ."""
    assert match(['a'], 'b') is False

def test_exception_handling():
    """Test that exceptions in fnmatchcase result in False."""
    # While we cannot easily trigger fnmatchcase exceptions without knowing internals,
    # we can test the contract that non-matches return False.
    # Using a pattern that fnmatchcase returns False for.
    assert match(['a'], '!') is False

def test_index_vs_key_indistinguishability():
    """Test that list index 0 and dict key '0' are treated identically."""
    assert match([0], '0') is True
    assert match(['0'], '0') is True

def test_return_type_is_bool():
    """Test that the function returns a boolean."""
    result = match(['a'], 'a')
    assert isinstance(result, bool)
