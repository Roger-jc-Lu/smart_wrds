import os
import re
import datetime
import json
import requests

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Streamlit page
st.set_page_config(page_title="Param Collector", layout="wide")
st.title("Stock Data Collection")

# OpenAI and LangChain setup 
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Chat init
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """
You are a parameter‑collection assistant.
Collect exactly four parameters from the user:
 • ticker (uppercase stock symbol)
 • start_date (YYYY-MM-DD)
 • end_date (YYYY-MM-DD)
 • granularity (one of: 1s,5s,15s,30s,1m,2m,5m,15m,30m,60m,1h,1d)

Ask follow‑up questions until you have all four.
Once you do, reply with pure JSON only, for example:

{"ticker":"AAPL","start_date":"2020-01-01","end_date":"2023-06-10","granularity":"1m"}

The app will also:
 1) Immediately check any bare YYYY-MM-DD you type to ensure ≤ today  
 2) Immediately check any bare uppercase ticker you type against Yahoo Finance  
If those checks fail, you’ll see an error message before the LLM is even called,
and the input will not be sent to the model.
"""
        }
    ]

# User input
user_input = st.text_input(
    "Enter your request, e.g.:  \n"
    "`Get AAPL data from 2020-01-01 to 2023-06-10 with granularity 1m`",
    key="user_input",
)
send = st.button("Send")

# Early validation on bare dates and tickers
if send and user_input:
    immediate_error = None
    txt = user_input.strip()

    # If exactly a date, ensure not in the future
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", txt):
        try:
            entered = datetime.datetime.strptime(txt, "%Y-%m-%d").date()
            if entered > datetime.date.today():
                immediate_error = "That date is in the future. Please enter a date ≤ today."
        except ValueError:
            immediate_error = "Invalid date format. Use YYYY-MM-DD."

    # verify via Yahoo Finance chart API
    elif re.fullmatch(r"[A-Z]{1,5}", txt):
        symbol = txt
        chart_url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            "?range=1d&interval=1d"
        )
        try:
            resp = requests.get(chart_url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            chart = data.get("chart", {})
            # only reject if Yahoo explicitly signals an error or returns no result
            if chart.get("error") or not chart.get("result"):
                immediate_error = "Invalid ticker symbol. Please enter a real ticker."
        except requests.RequestException:
            # network or transient error
            pass

    # If early validation failed, show error and do not call the LLM
    if immediate_error:
        st.error(immediate_error)
    else:
        # Append user message and call LangChain
        st.session_state.messages.append({"role": "user", "content": user_input})
        if not OPENAI_KEY:
            assistant_reply = "Please enter a valid OpenAI API Key in the sidebar."
        else:
            lc_msgs = []
            for m in st.session_state.messages:
                if m["role"] == "system":
                    lc_msgs.append(SystemMessage(content=m["content"]))
                elif m["role"] == "user":
                    lc_msgs.append(HumanMessage(content=m["content"]))
                else:
                    lc_msgs.append(AIMessage(content=m["content"]))

            llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
            ai_out = llm(lc_msgs)
            assistant_reply = ai_out.content

        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# Give the conversation
for msg in st.session_state.messages[1:]:
    who = "**User**" if msg["role"] == "user" else "**Assistant**"
    st.markdown(f"{who}: {msg['content']}")

# Final JSON parse and downstream validation
last = st.session_state.messages[-1]["content"]
try:
    params = json.loads(last)
    required = {"ticker", "start_date", "end_date", "granularity"}
    if required.issubset(params):
        # end_date ≤ today
        end_dt = datetime.datetime.strptime(params["end_date"], "%Y-%m-%d").date()
        if end_dt > datetime.date.today():
            st.error("End date is in the future. Please re‑enter End date.")
            st.session_state.messages.pop()
        else:
            # ticker exists via Yahoo chart API
            sym = params["ticker"].upper()
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                "?range=1d&interval=1d"
            )
            resp = requests.get(url, timeout=5).json()
            chart = resp.get("chart", {})
            if chart.get("error") or not chart.get("result"):
                st.error("Invalid ticker name. Please re‑enter ticker.")
                st.session_state.messages.pop()
            else:
                st.success(f"Parameters collected: {json.dumps(params)}")
                # TODO: proceed with downstream data fetching
except json.JSONDecodeError:
    pass