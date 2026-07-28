    def test_new_last_node_no_exceptions():
        node = SentinelNode()
        result = node.new_last_node()
        assert result is not None
    