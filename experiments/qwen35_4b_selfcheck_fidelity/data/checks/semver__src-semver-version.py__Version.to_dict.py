import pytest
import semver

def test_to_dict_returns_dict():
    """Asserts that to_dict returns a dictionary."""
    v = semver.Version(3, 2, 1)
    result = v.to_dict()
    assert isinstance(result, dict)

def test_to_dict_has_expected_keys():
    """Asserts that the returned dictionary contains the keys documented in the example."""
    v = semver.Version(3, 2, 1)
    result = v.to_dict()
    expected_keys = {'major', 'minor', 'patch', 'prerelease', 'build'}
    assert set(result.keys()) == expected_keys

def test_to_dict_major_matches_major():
    """Asserts that the 'major' key matches the major version number."""
    v = semver.Version(3, 2, 1)
    result = v.to_dict()
    assert result['major'] == 3

def test_to_dict_minor_matches_minor():
    """Asserts that the 'minor' key matches the minor version number."""
    v = semver.Version(3, 2, 1)
    result = v.to_dict()
    assert result['minor'] == 2

def test_to_dict_patch_matches_patch():
    """Asserts that the 'patch' key matches the patch version number."""
    v = semver.Version(3, 2, 1)
    result = v.to_dict()
    assert result['patch'] == 1

def test_to_dict_prerelease_is_none_by_default():
    """Asserts that 'prerelease' is None when not specified."""
    v = semver.Version(3, 2, 1)
    result = v.to_dict()
    assert result['prerelease'] is None

def test_to_dict_build_is_none_by_default():
    """Asserts that 'build' is None when not specified."""
    v = semver.Version(3, 2, 1)
    result = v.to_dict()
    assert result['build'] is None

def test_to_dict_values_are_integers():
    """Asserts that version components are integers."""
    v = semver.Version(3, 2, 1)
    result = v.to_dict()
    assert isinstance(result['major'], int)
    assert isinstance(result['minor'], int)
    assert isinstance(result['patch'], int)
