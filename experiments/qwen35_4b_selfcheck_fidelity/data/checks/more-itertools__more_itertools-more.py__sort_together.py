def test_sort_together_key_list_multi():
    """Test sorting based on multiple indices using key_list."""
    iterables = [[1, 3, 2], ['a', 'b', 'c']]
    result = sort_together(iterables, key_list=[0, 1])
    # Sort by (1, 'a'), (3, 'b'), (2, 'c')
    # Order: (1, 'a'), (2, 'c'), (3, 'b')
    # Result: [[1, 2, 3], ['a', 'c', 'b']]
    assert result == [[1, 2, 3], ['a', 'c', 'b']]
