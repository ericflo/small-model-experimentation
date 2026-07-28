    import pytest
    from dpath.segments import make_walkable

    def test_make_walkable_dict_items():
        """Test that dict nodes yield key-value pairs via items()."""
        node = {"a": 1, "b": 2}
        result = list(make_walkable(node))
        assert result == [("a", 1), ("b", 2)]

    def test_make_walkable_sequence_indexed():
        """Test that sequence nodes yield index-value pairs via zip."""
        node = [10, 20, 30]
        result = list(make_walkable(node))
        assert result == [(0, 10), (1, 20), (2, 30)]

    def test_make_walkable_tuple_sequence():
        """Test that tuple nodes behave like sequence nodes."""
        node = (1, 2)
        result = list(make_walkable(node))
        assert result == [(0, 1), (1, 2)]

    def test_make_walkable_empty_dict():
        """Test that empty dict nodes result in an empty iterator."""
        node = {}
        result = list(make_walkable(node))
        assert result == []

    def test_make_walkable_empty_list():
        """Test that empty sequence nodes result in an empty iterator."""
        node = []
        result = list(make_walkable(node))
        assert result == []

    def test_make_walkable_non_iterable_edge_case():
        """Test that non-dict/sequence edge cases result in an empty iterator."""
        node = 42
        result = list(make_walkable(node))
        assert result == []

    def test_make_walkable_returns_iterator():
        """Test that the function returns an iterator object."""
        node = {"a": 1}
        result = make_walkable(node)
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_make_walkable_yielded_items_are_tuples():
        """Test that yielded items are tuples of length 2."""
        node = {"a": 1}
        result = list(make_walkable(node))
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_make_walkable_sequence_index_starts_at_zero():
        """Test that sequence indices start at zero."""
        node = [1, 2]
        result = list(make_walkable(node))
        assert result[0][0] == 0

    def test_make_walkable_dict_keys_preserved():
        """Test that dict keys are preserved as the first element of the tuple."""
        node = {"x": "y"}
        result = list(make_walkable(node))
        assert result[0][0] == "x"
    