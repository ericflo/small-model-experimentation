    import pytest
    from glom import flatten
    import types

    def test_basic_flattening():
        target = [[1, 2], [3], [4]]
        result = flatten(target)
        assert result == [1, 2, 3, 4]

    def test_init_tuple():
        target = [[1, 2], [3]]
        result = flatten(target, init=tuple)
        assert result == (1, 2, 3)

    def test_init_int_levels():
        target = [[1, 2], [3], [4]]
        result = flatten(target, init=int, levels=2)
        assert result == 10

    def test_init_lazy():
        target = [[1, 2], [3]]
        result = flatten(target, init="lazy")
        assert isinstance(result, types.GeneratorType)

    def test_mixed_iterables():
        list_of_iterables = [range(2), [2, 3], (4, 5)]
        result = flatten(list_of_iterables)
        assert result == [0, 1, 2, 3, 4, 5]

    def test_non_iterable_error():
        target = 10
        with pytest.raises(Exception) as exc_info:
            flatten(target)
        assert "iterable" in str(exc_info.value)
        assert "int" in str(exc_info.value)

    def test_levels_default():
        target = [[1, 2], [3]]
        result = flatten(target)
        assert result == [1, 2, 3]

    def test_levels_custom():
        target = [[[1, 2], [3]]]
        result = flatten(target, levels=1)
        assert result == [[1, 2], [3]]
        result2 = flatten(target, levels=2)
        assert result2 == [1, 2, 3]
    