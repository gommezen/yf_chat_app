# tests/test_core.py
import time

#from urllib.parse import urlparse
import app


def test_parse_slash_price_defaults():
    plan = app.parse_slash("/price AAPL")
    assert plan["action"] == "price"
    assert plan["args"]["ticker"] == "AAPL"
    # dates should be filled by default
    assert "start" in plan["args"] and "end" in plan["args"]
    assert plan["args"]["interval"] == "1d"


def test_parse_slash_compare_with_params():
    plan = app.parse_slash("/compare TSLA F 2024-03-01 2024-09-20 1d")
    assert plan["action"] == "compare"
    assert plan["args"]["tickers"] == ["TSLA", "F"]
    assert plan["args"]["start"] == "2024-03-01"
    assert plan["args"]["end"] == "2024-09-20"
    assert plan["args"]["interval"] == "1d"


def test_normalize_news_item_basic():
    raw = {
        "title": "Sample Headline",
        "publisher": "Example News",
        "summary": "Hello world",
        "providerPublishTime": int(time.time()),
        "link": "https://finance.yahoo.com/news/sample-headline-abc123.html",
    }
    n = app.normalize_news_item(raw)
    assert n["title"] == "Sample Headline"
    assert n["publisher"] == "Example News"
    assert n["summary"] == "Hello world"
    assert n["link"].startswith("https://")
    # timestamp formatting is handled elsewhere; just make sure it exists
    assert n["ts"]


def test_fmt_ts_accepts_unix_and_iso():
    unix = int(time.time())
    s1 = app._fmt_ts(unix)
    assert "UTC" in s1

    iso = "2025-09-20T19:01:44Z"
    s2 = app._fmt_ts(iso)
    assert "UTC" in s2


def test_slug_from_url_fallback():
    url = "https://finance.yahoo.com/news/amazon-google-microsoft-reportedly-warn-190144778.html"
    slug = app._slug_from_url(url)
    # Basic quality check: starts with capitalized word
    assert slug[0].isupper()
    # Should not be empty
    assert slug
