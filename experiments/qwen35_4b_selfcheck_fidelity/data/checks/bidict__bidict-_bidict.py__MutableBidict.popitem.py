    import pytest
    from bidict import MutableBidict

    def test_popitem_returns_tuple():
        """Asserts popitem returns a tuple as per signature."""
        b = MutableBidict()
        b[1] = 10
        result = b.popitem()
        assert isinstance(result, tuple)

    def test_popitem_returns_two_elements():
        """Asserts popitem returns exactly two elements (key, value)."""
        b = MutableBidict()
        b[1] = 10
        result = b.popitem()
        assert len(result) == 2

    def test_popitem_removes_item():
        """Asserts popitem removes the item from the dictionary."""
        b = MutableBidict()
        b[1] = 10
        b.popitem()
        assert 1 not in b

    def test_popitem_raises_keyerror_on_empty():
        """Asserts popitem raises KeyError if the dictionary is empty."""
        b = MutableBidict()
        with pytest.raises(KeyError):
            b.popitem()

    def test_popitem_returns_key_value_pair():
        """Asserts popitem returns the key and value of the removed item."""
        b = MutableBidict()
        b[1] = 10
        result = b.popitem()
        assert result[0] == 1
        assert result[1] == 10

    def test_popitem_preserves_other_items():
        """Asserts popitem does not remove items other than the one popped."""
        b = MutableBidict()
        b[1] = 10
        b[2] = 20
        b.popitem()
        assert 2 in b
        assert b[2] == 20

    def test_popitem_multiple_calls():
        """Asserts popitem can be called multiple times until empty."""
        b = MutableBidict()
        b[1] = 10
        b[2] = 20
        b[3] = 30
        r1 = b.popitem()
        r2 = b.popitem()
        r3 = b.popitem()
        assert len(b) == 0
        assert r1[0] in [1, 2, 3]
        assert r2[0] in [1, 2, 3]
        assert r3[0] in [1, 2, 3]
    