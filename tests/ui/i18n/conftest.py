import pytest
from django.utils import translation


@pytest.fixture(autouse=True)
def reset_language():
    """Each test starts in English."""
    translation.activate("en")
    yield
    translation.deactivate_all()
