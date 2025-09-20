# Ticker Talk

Chat with yfinance to fetch 
**quotes, history, dividends, splits, metadata, and news**.  

Use slash-commands (deterministic) or enable an **LLM planner** for natural language via **Ollama (local, free)** or **OpenAI**.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Slash commands
- `/price TICKER [START END [INTERVAL]]` — e.g. `/price AAPL 2024-01-01 2024-09-20 1d`
- `/dividends TICKER`
- `/splits TICKER`
- `/info TICKER`
- `/news TICKER`

### Use a free local LLM (Ollama)
1. Install Ollama: https://ollama.com
2. Pull a model and run the server:
   ```bash
   ollama pull llama3.1:8b-instruct-q8_0
   ollama serve
   ```
3. In the Streamlit sidebar: toggle **Use LLM planner** → choose **Ollama (local)**.

### OpenAI (optional)
Create `.env` with:
```
OPENAI_API_KEY=sk-...
```
Then choose **OpenAI** in the sidebar.

### Notes
- Data from yfinance/Yahoo Finance may be delayed or incomplete.
