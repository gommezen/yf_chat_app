# tests/test_smoke.py


def test_env_imports():
    import pandas  # noqa: F401
    import plotly  # noqa: F401
    import requests  # noqa: F401
    import streamlit  # noqa: F401
    import yfinance  # noqa: F401


def test_python_sanity():
    assert 2 + 2 == 4
