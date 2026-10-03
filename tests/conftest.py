import pytest

import lessPy


@pytest.fixture(autouse=True)
def _default_style():
    # each test starts, and leaves, in the default theme, so a style()
    # set by one test cannot reach another
    lessPy.style()
    yield
    lessPy.style()
