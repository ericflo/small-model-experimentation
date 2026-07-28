    import pytest
    from voluptuous import Schema, Email, MultipleInvalid

    def test_valid_email():
        s = Schema(Email())
        result = s('t@x.com')
        assert result == 't@x.com'

    def test_invalid_no_at():
        s = Schema(Email())
        with pytest.raises(MultipleInvalid) as exc_info:
            s("a.com")
        assert 'expected an email address' in str(exc_info.value)

    def test_invalid_double_dot():
        s = Schema(Email())
        with pytest.raises(MultipleInvalid) as exc_info:
            s("a@.com")
        assert 'expected an email address' in str(exc_info.value)

    def test_error_message():
        s = Schema(Email())
        with pytest.raises(MultipleInvalid) as exc_info:
            s("a.com")
        assert 'expected an email address' in str(exc_info.value)

    def test_empty_string():
        s = Schema(Email())
        with pytest.raises(MultipleInvalid) as exc_info:
            s("")
        assert 'expected an email address' in str(exc_info.value)

    def test_none_value():
        s = Schema(Email())
        with pytest.raises(MultipleInvalid) as exc_info:
            s(None)
        assert 'expected an email address' in str(exc_info.value)
    