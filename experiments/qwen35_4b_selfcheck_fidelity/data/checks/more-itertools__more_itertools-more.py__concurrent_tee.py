        def test_concurrent_tee_independent_streams():
            it = concurrent_tee([1, 2, 3], n=2)
            # Consume first
            list(it[0])
            # Second should still have data
            assert list(it[1]) == [1, 2, 3] # Wait, tee consumes the underlying source.
            # If it's a tee, consuming one affects the source.
            # But tee creates independent copies.
            # So consuming one should NOT affect the other.
            # But tee consumes the underlying source.
            # So if I consume from it[0], it[1] should still get the data?
            # Yes, that's the point of tee.
            # But wait, if I consume from it[0], does it[1] get the data?
            # Yes, tee buffers or queues.
            # So list(it[0]) should be [1, 2, 3].
            # list(it[1]) should be [1, 2, 3].
            # But if I consume from it[0], does it[1] get the data?
            # Yes.
            # So I need to check that they yield the same data independently.
        