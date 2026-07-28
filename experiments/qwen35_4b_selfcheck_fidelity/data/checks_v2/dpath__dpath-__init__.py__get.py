    import pytest
    from dpath import get
    from collections.abc import MutableMapping

    def test_basic_success():
        """Test that a single match returns the value."""
        obj = {"a": 1}
        result = get(obj, "a")
        assert result == 1

    def test_no_match_with_default():
        """Test that a missing glob returns the default value."""
        obj = {"a": 1}
        result = get(obj, "b", default="default_val")
        assert result == "default_val"

    def test_no_match_without_default():
        """Test that a missing glob raises KeyError when no default is provided."""
        obj = {"a": 1}
        with pytest.raises(KeyError):
            get(obj, "b")

    def test_multiple_matches_raises_valueerror():
        """Test that multiple matches raise ValueError."""
        # Construct a scenario where 'a' matches both 'a' and 'a,b'
        # based on the separator logic implied by the signature.
        obj = {"a": 1, "a,b": 2}
        with pytest.raises(ValueError):
            get(obj, "a", separator=",")

    def test_empty_mapping():
        """Test that an empty mapping raises KeyError for any glob."""
        obj = {}
        with pytest.raises(KeyError):
            get(obj, "a")

    def test_default_separator():
        """Test that the default separator is used when not specified."""
        obj = {"a,b": 1}
        result = get(obj, "a,b")
        assert result == 1

    def test_separator_argument():
        """Test that the separator argument is accepted and used."""
        obj = {"a,b": 1}
        result = get(obj, "a,b", separator=",")
        assert result == 1
    