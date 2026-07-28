    import pytest
    from voluptuous import Schema, MultipleInvalid
    from voluptuous.validators import Url, UrlInvalid

    def test_valid_url_passes():
        s = Schema(Url())
        result = s('http://w3.org')
        assert result == 'http://w3.org'

    def test_invalid_int_raises():
        s = Schema(Url())
        with pytest.raises(MultipleInvalid):
            s(1)

    def test_error_message():
        s = Schema(Url())
        with pytest.raises(MultipleInvalid) as exc_info:
            s(1)
        assert str(exc_info.value) == 'expected a URL'

    def test_schema_creation():
        s = Schema(Url())
        assert isinstance(s, Schema)

    def test_url_invalid_class():
        assert UrlInvalid is not None

    def test_non_string_invalid():
        s = Schema(Url())
        with pytest.raises(MultipleInvalid):
            s(None)

    def test_url_with_port():
        s = Schema(Url())
        result = s('http://w3.org:80')
        assert result == 'http://w3.org:80'
    