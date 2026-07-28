    import pytest
    from more_itertools import nth_combination

    def test_basic_example():
        """Test the example provided in the docstring."""
        result = nth_combination(range(5), 3, 5)
        assert result == (0, 3, 4)

    def test_negative_r_raises_value_error():
        """Test that negative r raises ValueError as per docstring."""
        with pytest.raises(ValueError):
            nth_combination(range(5), -1, 0)

    def test_invalid_index_raises_index_error():
        """Test that an index out of bounds raises IndexError as per docstring."""
        with pytest.raises(IndexError):
            nth_combination(range(5), 2, 100)

    def test_negative_index_raises_index_error():
        """Test that a negative index raises IndexError as per docstring."""
        with pytest.raises(IndexError):
            nth_combination(range(5), 2, -1)

    def test_empty_iterable_r_zero():
        """Test empty iterable with r=0 returns empty tuple."""
        result = nth_combination([], 0, 0)
        assert result == ()

    def test_empty_iterable_r_nonzero():
        """Test empty iterable with r>0 raises IndexError."""
        with pytest.raises(IndexError):
            nth_combination([], 1, 0)

    def test_r_equals_length():
        """Test r equals iterable length returns the single combination."""
        result = nth_combination([1, 2], 2, 0)
        assert result == (1, 2)

    def test_r_equals_one():
        """Test r equals 1 returns elements in order."""
        result = nth_combination([1, 2, 3], 1, 1)
        assert result == (2,)

    def test_return_type_is_tuple():
        """Test that the result is a tuple, consistent with combinations output."""
        result = nth_combination(range(5), 3, 0)
        assert isinstance(result, tuple)
    