import pytest
from django.utils import translation


@pytest.fixture(autouse=True)
def reset_language():
    """Each test starts from Django's default language and restores it after,
    so activate()/set_language side effects never leak between tests."""
    translation.activate("en")
    yield
    translation.deactivate_all()
