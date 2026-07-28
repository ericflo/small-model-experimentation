    import pytest
    from more_itertools import tail
    import types

    def test_tail_basic_string():
        """Test basic usage matching docstring example."""
        t = tail(3, 'ABCDEFG')
        assert list(t) == ['E', 'F', 'G']

    def test_tail_empty_iterable():
        """Test behavior with an empty iterable."""
        t = tail(3, [])
        assert list(t) == []

    def test_tail_n_zero():
        """Test behavior when n is zero."""
        t = tail(0, 'ABCDEFG')
        assert list(t) == []

    def test_tail_n_exceeds_length():
        """Test behavior when n is larger than iterable length."""
        t = tail(10, 'ABC')
        assert list(t) == ['A', 'B', 'C']

    def test_tail_n_equals_length():
        """Test behavior when n equals iterable length."""
        t = tail(3, 'ABC')
        assert list(t) == ['A', 'B', 'C']

    def test_tail_returns_iterator():
        """Test that the function returns an iterator, not a list."""
        t = tail(3, 'ABCDEFG')
        assert isinstance(t, types.Iterator)

    def test_tail_list_input():
        """Test behavior with a list input."""
        t = tail(2, [1, 2, 3, 4, 5])
        assert list(t) == [4, 5]

    def test_tail_generator_input():
        """Test behavior with a generator input."""
        def gen():
            yield 1
            yield 2
            yield 3
            yield 4
            yield 5
        t = tail(2, gen())
        assert list(t) == [4, 5]
    