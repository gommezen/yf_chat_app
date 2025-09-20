# app.py
# Streamlit chat app to talk to yfinance: quotes, history, dividends, splits, metadata, and news.
# Optional: natural-language queries via OpenAI or Ollama (local, free). Otherwise supports simple slash-commands.

import os
import json
import re
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import requests

# Optional OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

st.set_page_config(page_title="TickerTalk", page_icon="📈", layout="wide")

# ------------- Helpers & cache -------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
    # Normalize columns
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

# ------------- Utilities -------------
def normalize_ticker_arg(v):
    """Ensure ticker is a string symbol. If list is provided, return first element and a note."""
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

# ------------- LLM providers (optional) -------------
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
    "Example output: {\"action\":\"price\",\"args\":{\"ticker\":\"AAPL\",\"start\":\"2024-01-01\",\"end\":\"2024-12-31\",\"interval\":\"1d\"},\"explanation\":\"Price history requested\"}"
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

INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1d","5d","1wk","1mo","3mo"}

def one_year_range():
    today = date.today()
    return (today - timedelta(days=365)).isoformat(), (today).isoformat()

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
        left = text.find('{'); right = text.rfind('}')
        if left != -1 and right != -1:
            text = text[left:right+1]
        return json.loads(text)
    except Exception:
        return None


def plan_with_ollama(message: str):
    # Requires Ollama running locally or remotely. Uses /api/chat.
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
        left = text.find('{'); right = text.rfind('}')
        if left != -1 and right != -1:
            text = text[left:right+1]
        return json.loads(text)
    except Exception:
        return None

# ------------- UI -------------
st.title("📈 TickerTalk")
st.caption("Query stock data (quotes, history, dividends, splits, metadata, news). Optional natural language via OpenAI or local Ollama.")

with st.sidebar:
    st.subheader("How to use")
    st.markdown(SLASH_HELP)
    st.info("Tip: Toggle an LLM planner below for natural questions like 'compare TSLA and F last 6 months'.")
    st.warning("Data via yfinance/Yahoo Finance for educational purposes only. Not investment advice.")

    st.divider()
    st.subheader("LLM Settings")
    use_llm = st.toggle("Use LLM planner", value=False, help="If off, only slash-commands are parsed.")
    provider = st.radio("Provider", ["None","Ollama (local)","OpenAI"], index=0, horizontal=False)
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

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
user_msg = st.chat_input("Ask about a ticker… e.g., /compare TSLA F 2024-03-01 2024-09-20 1d")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Decide a plan
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
                    st.write(f"**{ticker.upper()}** price from **{start}** to **{end}** at **{interval}** interval.")
                    fig = px.line(df, x="date", y="Close", title=f"{ticker.upper()} Close Price")
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Show raw OHLCV"):
                        st.dataframe(df)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, file_name=f"{ticker.upper()}_{start}_to_{end}_{interval}.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)

        elif action == "compare":
            tickers = args.get("tickers", [])
            start = args.get("start")
            end = args.get("end")
            interval = args.get("interval", "1d")
            normalize = st.checkbox("Normalize to 100 at start", value=True)
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
                        plot_df = df.sort_values(["Ticker", "date"]).copy()
                        if normalize:
                            first_close = plot_df.groupby("Ticker")["Close"].transform("first")
                            first_close = first_close.replace(0, pd.NA)  # safety
                            plot_df["Close_norm"] = (plot_df["Close"] / first_close) * 100
                            y_col, y_title = "Close_norm", "Indexed to 100"
                        else:
                            y_col, y_title = "Close", "Close"
                            
                        fig = px.line(plot_df, x="date", y=y_col, color="Ticker", title=f"{t1} vs {t2} {y_title}")
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("Show raw merged data"):
                            st.dataframe(df)
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download merged CSV", csv, file_name=f"{t1}_{t2}_{start}_to_{end}_{interval}.csv", mime="text/csv")
                except Exception as e:
                    st.exception(e)

        elif action == "dividends":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            try:
                df = fetch_dividends(ticker.upper())
                if df.empty:
                    st.info(f"No dividend data for {ticker.upper()}.")
                else:
                    st.write(f"**{ticker.upper()}** dividends")
                    fig = px.bar(df, x="date", y="dividend", title=f"{ticker.upper()} Dividends")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, file_name=f"{ticker.upper()}_dividends.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)

        elif action == "splits":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            try:
                df = fetch_splits(ticker.upper())
                if df.empty:
                    st.info(f"No split data for {ticker.upper()}.")
                else:
                    st.write(f"**{ticker.upper()}** stock splits")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, file_name=f"{ticker.upper()}_splits.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)

        elif action == "info":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            try:
                info = fetch_info(ticker.upper())
                if not info:
                    st.info(f"No info for {ticker.upper()}.")
                else:
                    name = info.get("longName") or info.get("shortName") or ticker.upper()
                    st.subheader(f"{name} ({ticker.upper()})")
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
            except Exception as e:
                st.exception(e)

        elif action == "news":
            raw_ticker = args.get("ticker", "")
            ticker, note = normalize_ticker_arg(raw_ticker)
            if note:
                st.info(note)
            try:
                news = fetch_news(ticker.upper())
                if not news:
                    st.info(f"No news for {ticker.upper()}.")
                else:
                    st.write(f"Latest news for **{ticker.upper()}** (from Yahoo Finance)")
                    for item in news:
                        st.markdown(f"### [{item.get('title','(no title)')}]({item.get('link','#')})")
                        st.caption(item.get('publisher',''))
                        st.write(item.get('summary',''))
                        st.divider()
            except Exception as e:
                st.exception(e)

        else:
            if llm_warning:
                st.warning(llm_warning)
            st.markdown("I can help with these:")
            st.markdown(SLASH_HELP)

# ------------- Footer -------------
st.write("\n\n—\n**Disclaimer**: This app uses yfinance (which scrapes Yahoo Finance). Data may be delayed or inaccurate. Educational use only, not financial advice.")


# --------------------
# requirements.txt (place in the same folder)
# --------------------
# streamlit
# yfinance
# pandas
# plotly
# python-dotenv
# openai
# requests


# --------------------
# .env (create this file; optional)
# --------------------
# OPENAI_API_KEY=sk-your-key-here
# OLLAMA_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.1:8b-instruct-q8_0
