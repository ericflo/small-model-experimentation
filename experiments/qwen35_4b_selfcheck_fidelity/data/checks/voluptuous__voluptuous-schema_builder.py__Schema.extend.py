    import pytest
    from voluptuous import Schema

    def test_extend_returns_new_schema():
        base = Schema({'a': 1})
        extended = base.extend({'b': 2})
        assert extended is not base

    def test_extend_does_not_modify_self():
        base = Schema({'a': 1})
        base.extend({'b': 2})
        assert base == Schema({'a': 1}) # Or check attributes

    def test_extend_does_not_modify_input_schema():
        input_schema = Schema({'a': 1})
        base = Schema({'b': 2})
        base.extend(input_schema)
        assert input_schema == Schema({'a': 1})

    def test_extend_inherits_required():
        base = Schema({'a': 1}, required=True)
        extended = base.extend({'b': 2})
        # Need to check if extended has required=True
        # Since I don't know the internal structure, I'll check the result is a Schema
        # and rely on the contract that it inherits.
        # Actually, to test inheritance without knowing internal attributes,
        # I can check if the behavior matches the docstring implication.
        # But I can't easily test the *value* of required on the result without validation.
        # However, the contract says "inherits... unless overridden".
        # I can test that passing required=True overrides.
        # I can test that passing required=None (default) inherits.
        # To verify inheritance, I might need to validate the result against a schema that expects the inherited constraint.
        # But I can't validate the result's constraints easily without knowing the internal structure.
        # Let's stick to the explicit contract assertions:
        # 1. Return type is Schema.
        # 2. If required is passed, it overrides.
        # 3. If required is not passed, it inherits.
        # Since I can't introspect the result's `required` attribute easily without knowing the class structure,
        # I will focus on the behavior described: "overrides... unless overridden".
        # I will test that passing `required=True` results in a schema that enforces it (if I could).
        # But I can't run validation logic on the result easily without knowing the schema structure.
        # Let's assume the test can check the return value type.
        # Actually, I can check if the result is a Schema instance.
        # For inheritance, I will test that the default behavior (not passing required) works without error,
        # and that passing required=True works.
        # To be strict about "inherits", I might need to check the attributes if I can import the class.
        # But I can't assume I know the class attributes.
        # Let's focus on the explicit overrides and the return type.
        # Wait, I can check `extended.required` if I assume `Schema` exposes it.
        # The docstring says "inherits the required... parameters of this".
        # I will assume `Schema` objects have a `required` attribute.
        # This is a reasonable assumption based on the docstring mentioning "parameters of this Schema".
        