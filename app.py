# app.py
# TickerTalk — Streamlit chat app to talk to yfinance: quotes, history, dividends, splits, metadata, and news.
# Optional: natural-language queries via OpenAI or Ollama (local, free). Otherwise supports simple slash-commands.

import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# Optional OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()
st.set_page_config(page_title="TickerTalk", page_icon="📈", layout="wide")

# ---------------- Helpers & cache ----------------
def _get_nested(d, *path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _slug_from_url(url: str) -> str:
    try:
        path = urlparse(url).path.strip("/").split("/")[-1]
        slug = path.replace("-", " ").strip()
        return slug[:1].upper() + slug[1:] if slug else "(untitled)"
    except Exception:
        return "(untitled)"

def _fmt_ts(ts) -> str:
    if ts is None or ts == "":
        return ""
    # Try unix seconds
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    # Try ISO-8601 string like "2025-09-20T19:01:44Z"
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ""

def normalize_news_item(item: dict) -> dict:
    link = (
        item.get("link")
        or item.get("url")
        or _get_nested(item, "canonicalUrl", "url")
        or _get_nested(item, "clickThroughUrl", "url")
        or "#"
    )
    title = item.get("title") or item.get("headline") or ""
    publisher = (
        item.get("publisher")
        or _get_nested(item, "provider", "displayName")
        or (urlparse(link).netloc if link and link != "#" else "")
    )
    ts = (
        item.get("providerPublishTime")
        or item.get("published")
        or item.get("pubDate")
        or item.get("displayTime")
        or ""
    )
    summary = item.get("summary") or item.get("description") or item.get("content") or ""
    return {"link": link, "title": title, "publisher": publisher, "ts": ts, "summary": summary}

@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
    if not df.empty:
        df = df.reset_index().rename(columns={"Date": "date"})
    return df

@st.cache_data(show_spinner=False)
def fetch_dividends(ticker: str) -> pd.DataFrame:
    s = yf.Ticker(ticker).dividends
    df = s.reset_index()
    df.columns = ["date", "dividend"]
    return df

@st.cache_data(show_spinner=False)
def fetch_splits(ticker: str) -> pd.DataFrame:
    s = yf.Ticker(ticker).splits
    df = s.reset_index()
    df.columns = ["date", "split_ratio"]
    return df

@st.cache_data(show_spinner=False)
def fetch_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info or {}

@st.cache_data(show_spinner=False)
def fetch_news(ticker: str) -> list:
    return yf.Ticker(ticker).news or []

# ---------------- Utilities ----------------
def normalize_ticker_arg(v):
    note = None
    ticker = None
    if isinstance(v, str):
        ticker = v
    elif isinstance(v, (list, tuple)) and v:
        ticker = str(v[0])
        note = f"Multiple tickers provided ({', '.join(map(str, v))}); showing first: {ticker}. Use /compare for multi-ticker charts."
    elif v is not None:
        ticker = str(v)
    return ticker, note

# ---------------- LLM providers (optional) ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q8_0")

SYSTEM_PROMPT = (
    "You are a planning assistant for a stock chat app. "
    "Given a user message, output ONLY a compact JSON object with keys: action, args, explanation. "
    "Valid actions: price, dividends, splits, info, news, compare, help. "
    "For 'price', args must include ticker (string), start (YYYY-MM-DD), end (YYYY-MM-DD), interval (1d/1wk/1mo/60m/15m/5m/1m). "
    "For 'dividends'|'splits'|'info'|'news', args must include ticker. "
    "For 'compare', args must include tickers (array of 2 symbols), start, end, interval. "
    "If missing dates, default start to 1 year ago from today, end to today, interval to 1d. "
    "If message asks multiple things, choose the most central one. "
    'Example output: {"action":"price","args":{"ticker":"AAPL","start":"2024-01-01","end":"2024-12-31","interval":"1d"},"explanation":"Price history requested"}'
)

SLASH_HELP = (
    "**Slash commands**\n\n"
    "/price TICKER [START END [INTERVAL]]  → price chart. Example: `/price AAPL 2024-01-01 2024-09-20 1d`\n\n"
    "/dividends TICKER  → dividend history.\n\n"
    "/splits TICKER  → split history.\n\n"
    "/info TICKER  → company snapshot & metadata.\n\n"
    "/news TICKER  → latest Yahoo Finance news items.\n\n"
    "/compare T1 T2 [START END [INTERVAL]] → compare two tickers on one chart (optionally normalized).\n\n"
)

INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"}

def one_year_range():
    today = date.today()
    return (today - timedelta(days=365)).isoformat(), today.isoformat()

# Primitive parser for slash commands
slash_re = re.compile(r"^/(price|dividends|splits|info|news|compare)\s+(.+)$", re.I)

def parse_slash(message: str):
    m = slash_re.match(message.strip())
    if not m:
        return None
    action = m.group(1).lower()
    rest = m.group(2).strip()
    parts = rest.split()

    if action == "price":
        ticker = parts[0]
        start, end = one_year_range()
        interval = "1d"
        if len(parts) >= 3:
            start = parts[1]
            end = parts[2]
        if len(parts) >= 4 and parts[3] in INTERVALS:
            interval = parts[3]
        return {"action": "price", "args": {"ticker": ticker, "start": start, "end": end, "interval": interval}}

    if action == "compare":
        t1, t2 = parts[0], parts[1]
        start, end = one_year_range()
        interval = "1d"
        if len(parts) >= 4:
            start = parts[2]
            end = parts[3]
        if len(parts) >= 5 and parts[4] in INTERVALS:
            interval = parts[4]
        return {"action": "compare", "args": {"tickers": [t1, t2], "start": start, "end": end, "interval": interval}}

    else:
        ticker = parts[0]
        return {"action": action, "args": {"ticker": ticker}}

# ---- LLM planners ----
def plan_with_openai(message: str):
    if not client_openai:
        return None
    today = date.today().isoformat()
    start_default = (date.today() - timedelta(days=365)).isoformat()
    user_msg = f"TODAY={today}. If dates missing use start={start_default}, end={today}. Query: {message}"
    try:
        resp = client_openai.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        text = resp.output_text.strip()
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right != -1:
            text = text[left:right+1]
        return json.loads(text)
    except Exception:
        return None

def plan_with_ollama(message: str):
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "stream": False,
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = data.get("message", {}).get("content", "").strip()
        left = text.find("{")
        right = text.rfind("}")
        if left != -1 and right != -1:
            text = text[left:right+1]
        return json.loads(text)
    except Exception:
        return None

# ---------------- UI ----------------
st.title("📈 TickerTalk")
st.caption("Query stock data (quotes, history, dividends, splits, metadata, news). Optional natural language via OpenAI or local Ollama.")

with st.sidebar:
    st.subheader("How to use")
    st.markdown(SLASH_HELP)
    st.info("Tip: Toggle an LLM planner below for natural questions like 'compare TSLA and F last 6 months'.")
    st.warning("Data via yfinance/Yahoo Finance for educational purposes only. May be delayed or inaccurate.")

    st.divider()
    st.subheader("LLM Settings")
    use_llm = st.toggle("Use LLM planner", value=False, help="If off, only slash-commands are parsed.")
    provider = st.radio("Provider", ["None", "Ollama (local)", "OpenAI"], index=0, horizontal=False)
    if provider == "Ollama (local)":
        st.text_input("OLLAMA_URL", value=OLLAMA_URL, key="ollama_url_help")
        st.text_input("Model", value=OLLAMA_MODEL, key="ollama_model_help")
    elif provider == "OpenAI":
        st.text_input("OPENAI_API_KEY", value=(OPENAI_API_KEY[:6] + "…" if OPENAI_API_KEY else ""), type="password", help="Set in .env")

# Messages state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask for /price, /dividends, /splits, /info, /news, /compare — or enable the LLM planner in the sidebar to ask natural questions."}
    ]
# Which panel to show
if "active_view" not in st.session_state:
    st.session_state.active_view = None

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- Input & planner ----------------
user_msg = st.chat_input("Ask about a ticker… e.g., /compare TSLA F 2024-03-01 2024-09-20 1d")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    plan = parse_slash(user_msg)
    llm_warning = None
    if not plan and use_llm:
        if provider == "OpenAI":
            plan = plan_with_openai(user_msg)
            if plan is None:
                llm_warning = "OpenAI planner could not interpret your request (check API key/connectivity)."
        elif provider == "Ollama (local)":
            plan = plan_with_ollama(user_msg)
            if plan is None:
                llm_warning = "Ollama planner not reachable (is `ollama serve` running, model pulled?)."
    if not plan:
        plan = {"action": "help", "args": {}}

    action = plan.get("action")
    args = plan.get("args", {})

    with st.chat_message("assistant"):
        # ----------- PRICE -----------
        if action == "price":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            start = args.get("start")
            end = args.get("end")
            interval = args.get("interval", "1d")
            try:
                df = fetch_history(ticker.upper(), start, end, interval)
                if df.empty:
                    st.error("No data returned. Check ticker/dates/interval.")
                else:
                    st.session_state["last_price"] = {
                        "df": df,
                        "ticker": ticker.upper(),
                        "start": start,
                        "end": end,
                        "interval": interval,
                    }
                    st.session_state.active_view = "price"
                    st.success(f"Showing price for {ticker.upper()} ({start} → {end}, {interval}).")
            except Exception as e:
                st.exception(e)

        # ----------- COMPARE -----------
        elif action == "compare":
            tickers = args.get("tickers", [])
            start = args.get("start")
            end = args.get("end")
            interval = args.get("interval", "1d")
            if not (isinstance(tickers, (list, tuple)) and len(tickers) >= 2):
                st.error("Compare needs two tickers, e.g., /compare TSLA F 2024-03-01 2024-09-20 1d")
            else:
                t1, t2 = str(tickers[0]).upper(), str(tickers[1]).upper()
                try:
                    df1 = fetch_history(t1, start, end, interval)
                    df2 = fetch_history(t2, start, end, interval)
                    if df1.empty and df2.empty:
                        st.error("No data returned for either ticker.")
                    else:
                        df1["Ticker"], df2["Ticker"] = t1, t2
                        df = pd.concat([df1, df2], ignore_index=True)
                        st.session_state["last_compare"] = {
                            "df": df,
                            "tickers": [t1, t2],
                            "start": start,
                            "end": end,
                            "interval": interval,
                        }
                        st.session_state.active_view = "compare"
                        st.success(f"Showing comparison {t1} vs {t2}.")
                except Exception as e:
                    st.exception(e)

        # ----------- DIVIDENDS -----------
        elif action == "dividends":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            try:
                df = fetch_dividends(ticker.upper())
                st.session_state["last_dividends"] = {"df": df, "ticker": ticker.upper()}
                st.session_state.active_view = "dividends"
                st.success(f"Showing dividends for {ticker.upper()}.")
            except Exception as e:
                st.exception(e)

        # ----------- SPLITS -----------
        elif action == "splits":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            try:
                df = fetch_splits(ticker.upper())
                st.session_state["last_splits"] = {"df": df, "ticker": ticker.upper()}
                st.session_state.active_view = "splits"
                st.success(f"Showing splits for {ticker.upper()}.")
            except Exception as e:
                st.exception(e)

        # ----------- INFO -----------
        elif action == "info":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            try:
                info = fetch_info(ticker.upper())
                st.session_state["last_info"] = {"info": info, "ticker": ticker.upper()}
                st.session_state.active_view = "info"
                st.success(f"Showing company snapshot for {ticker.upper()}.")
            except Exception as e:
                st.exception(e)

        # ----------- NEWS -----------
        elif action == "news":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            try:
                news = fetch_news(ticker.upper())
                st.session_state["last_news"] = {"news": news, "ticker": ticker.upper()}
                st.session_state.active_view = "news"
                st.success(f"Showing news for {ticker.upper()}.")
            except Exception as e:
                st.exception(e)

        else:
            if llm_warning:
                st.warning(llm_warning)
            st.markdown("I can help with these:")
            st.markdown(SLASH_HELP)

# ---------------- Persistent renderer (ONLY active view) ----------------
view = st.session_state.active_view

if view == "price" and "last_price" in st.session_state:
    data = st.session_state["last_price"]
    df = data["df"]
    ticker = data["ticker"]
    start, end, interval = data["start"], data["end"], data["interval"]
    st.subheader(f"{ticker} Close Price")
    fig = px.line(df, x="date", y="Close", title=f"{ticker} Close Price")
    st.plotly_chart(fig, use_container_width=True, key="price_chart")
    with st.expander("Show raw OHLCV"):
        st.dataframe(df)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_{start}_to_{end}_{interval}.csv",
        mime="text/csv",
    )

elif view == "compare" and "last_compare" in st.session_state:
    data = st.session_state["last_compare"]
    df = data["df"]
    t1, t2 = data["tickers"]
    start, end, interval = data["start"], data["end"], data["interval"]

    normalize = st.checkbox("Normalize to 100 at start", value=True, key="compare_norm")
    plot_df = df.sort_values(["Ticker", "date"]).copy()
    if normalize:
        def first_non_null(s: pd.Series):
            s2 = s.dropna()
            return s2.iloc[0] if not s2.empty else pd.NA
        first_close = plot_df.groupby("Ticker")["Close"].transform(first_non_null)
        plot_df["Close_norm"] = (plot_df["Close"] / first_close) * 100
        if plot_df["Close_norm"].dropna().empty:
            st.warning("Normalization yielded no values (missing prices at the start). Showing raw Close instead.")
            y_col, y_title = "Close", "Close"
        else:
            y_col, y_title = "Close_norm", "Indexed to 100"
    else:
        y_col, y_title = "Close", "Close"

    st.subheader(f"{t1} vs {t2} {y_title}")
    fig = px.line(plot_df, x="date", y=y_col, color="Ticker", title=f"{t1} vs {t2} {y_title}")
    st.plotly_chart(fig, use_container_width=True, key="compare_chart")
    with st.expander("Show raw merged data"):
        st.dataframe(df)
    st.download_button(
        "Download merged CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{t1}_{t2}_{start}_to_{end}_{interval}.csv",
        mime="text/csv",
    )

elif view == "dividends" and "last_dividends" in st.session_state:
    data = st.session_state["last_dividends"]
    df = data["df"]
    ticker = data["ticker"]
    st.subheader(f"{ticker} Dividends")
    if df.empty:
        st.info(f"No dividend data for {ticker}.")
    else:
        fig = px.bar(df, x="date", y="dividend", title=f"{ticker} Dividends")
        st.plotly_chart(fig, use_container_width=True, key="div_chart")
        st.dataframe(df)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_dividends.csv",
            mime="text/csv",
        )

elif view == "splits" and "last_splits" in st.session_state:
    data = st.session_state["last_splits"]
    df = data["df"]
    ticker = data["ticker"]
    st.subheader(f"{ticker} Stock Splits")
    if df.empty:
        st.info(f"No split data for {ticker}.")
    else:
        st.dataframe(df)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_splits.csv",
            mime="text/csv",
        )

elif view == "info" and "last_info" in st.session_state:
    data = st.session_state["last_info"]
    info = data["info"]
    ticker = data["ticker"]
    st.subheader(f"{info.get('longName') or info.get('shortName') or ticker} ({ticker})")
    if not info:
        st.info(f"No info for {ticker}.")
    else:
        cols = st.columns(3)
        with cols[0]:
            st.metric("Current Price", info.get("currentPrice", "-"))
            st.metric("Market Cap", info.get("marketCap", "-"))
            st.metric("PE Ratio (TTM)", info.get("trailingPE", "-"))
        with cols[1]:
            st.metric("52w High", info.get("fiftyTwoWeekHigh", "-"))
            st.metric("52w Low", info.get("fiftyTwoWeekLow", "-"))
            st.metric("Dividend Yield", info.get("dividendYield", "-"))
        with cols[2]:
            st.metric("Beta", info.get("beta", "-"))
            st.metric("EPS (TTM)", info.get("trailingEps", "-"))
            st.metric("Sector", info.get("sector", "-"))
        st.write("\n**Business summary**")
        st.write(info.get("longBusinessSummary", "—"))
        with st.expander("Show raw metadata"):
            st.json(info)

elif view == "news" and "last_news" in st.session_state:
    data = st.session_state["last_news"]
    news = data.get("news") or []
    ticker = data.get("ticker", "").upper()
    st.subheader(f"Latest news for {ticker} (Yahoo Finance)")

    if not news:
        st.info(f"No news for {ticker}.")
    else:
        top_n = st.selectbox("Show top N items", [5, 10, 20, 50], index=1, key="news_topn")
        seen = set()
        for item in news[:top_n]:
            n = normalize_news_item(item)
            title = n["title"] or _slug_from_url(n["link"])
            link = n["link"]
            publisher = n["publisher"]
            when = _fmt_ts(n["ts"])
            summary = n["summary"]

            key = (title, link)
            if key in seen:
                continue
            seen.add(key)

            # Render (no raw dicts)
            st.markdown(f"### [{title}]({link})")
            meta = " · ".join([p for p in [publisher, when] if p])
            if meta:
                st.caption(meta)
            if summary:
                st.write(summary)
            st.divider()

# ---------------- Footer ----------------
st.write("\n\n—\n**Disclaimer**: This app uses yfinance (which scrapes Yahoo Finance). Data may be delayed or inaccurate. Educational use only, not financial advice.")
