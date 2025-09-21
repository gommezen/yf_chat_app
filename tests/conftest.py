# tests/conftest.py
import os

# Ensure Streamlit runs headless during tests to avoid any UI/telemetry weirdness
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "1")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
