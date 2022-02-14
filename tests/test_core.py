import pytest
from fatector.core import hi


def test_hi():
    assert hi().say_hi() == "Solid"
