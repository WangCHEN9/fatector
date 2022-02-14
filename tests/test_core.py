import pytest
from sample.core import hi


def test_hi():
    assert hi().say_hi() == "Solid"
