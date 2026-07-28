    import pytest
    from bidict._orderedbase import SentinelNode, Node

    def test_new_last_node_returns_node():
        node = SentinelNode()
        result = node.new_last_node()
        assert isinstance(result, Node)

    def test_new_last_node_returns_new_object():
        node = SentinelNode()
        result = node.new_last_node()
        assert result is not node

    def test_new_last_node_returns_non_none():
        node = SentinelNode()
        result = node.new_last_node()
        assert result is not None

    def test_new_last_node_multiple_calls_return_distinct():
        node = SentinelNode()
        result1 = node.new_last_node()
        result2 = node.new_last_node()
        assert result1 is not result2

    def test_new_last_node_no_arguments():
        node = SentinelNode()
        with pytest.raises(TypeError, match="takes no arguments"):
            node.new_last_node(1)

    def test_new_last_node_no_exceptions():
        node = SentinelNode()
        try:
            node.new_last_node()
        except Exception:
            pytest.fail("new_last_node raised an exception")

    def test_new_last_node_is_callable():
        node = SentinelNode()
        assert callable(node.new_last_node)

    def test_new_last_node_does_not_modify_self():
        node = SentinelNode()
        original_id = id(node)
        node.new_last_node()
        assert id(node) == original_id
    