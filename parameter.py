import os
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Streamlit page setup
st.set_page_config(page_title="Parameter Collector", layout="wide")
st.title("Stock Data Collection")

# OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_KEY:
    OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """
You are a parameter-collection assistant. Your job is to gather exactly four pieces of information from the user:
  • ticker: a stock symbol (any case). You must normalize it to uppercase and confirm its validity.
  • start_date: a date string. Users may enter in any reasonable format (e.g. "2020/1/5", "Jan 5, 2020", "20200105", "2020").
  • end_date: a date string, same flexibility as start_date.
  • granularity: one of [1s, 5s, 15s, 30s, 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d].

Requirements:
1. **Date parsing & validation**  
   - Parse any common date format the user provides.  
   - If only a year(e.g. "2015") is given as start_date, auto-fill missing month/day as "01-01"; If only a year(e.g. "2018") 
   is given as end_date, auto-fill missing month/day as "12-31".
   - If a year range (e.g. "2015-2019") is given, auto-fill start_date as "01-01" and end_date as "12-31".
   - If a year and month(e.g. "Jun 2015") is given, auto-fill missing day as "01".
   - Check that neither start_date nor end_date is in the future. If a future date is detected, set it to "2024-12-31"

2. **Granularity fallback**  
   - If the user’s granularity isn’t in the allowed list (e.g. "3m"), compute the nearest lower and higher valid options (e.g. "2m" and "5m") and ask "Would you like 2m or 5m?".

3. **Ticker confirmation**  
   - If ticker is in lowercase, it is valid and normalize the ticker to uppercase.  
   - If ticker looks invalid, suggest a similar ticker and ask "Did you mean X? (yes/no)".

Continue asking follow-up questions until you have valid values for all four fields.  
Once collected, respond **only** with pure JSON, for example:
{"ticker":"AAPL","start_date":"2020-01-01","end_date":"2023-06-10","granularity":"1m"}
"""
        }
    ]

# User input widget
user_input = st.text_input(
    "Enter your request, e.g.:  \n"
    "`Get AAPL data from 2020-01-01 to 2024-12-31 with granularity 5m`",
    key="user_input",
)
if st.button("Send") and user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Ensure we have an API key
    if not OPENAI_KEY:
        assistant_reply = "Please enter a valid OpenAI API Key in the sidebar."
    else:
        # Build LangChain messages
        lc_msgs = []
        for m in st.session_state.messages:
            if m["role"] == "system":
                lc_msgs.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_msgs.append(HumanMessage(content=m["content"]))
            else:
                lc_msgs.append(AIMessage(content=m["content"]))

        # Call the model
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        ai_out = llm(lc_msgs)
        assistant_reply = ai_out.content

    # Store and display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# Render the conversation
for msg in st.session_state.messages[1:]:
    speaker = "**User**" if msg["role"] == "user" else "**Assistant**"
    st.markdown(f"{speaker}: {msg['content']}")

# Check for final JSON output
last = st.session_state.messages[-1]["content"]
try:
    params = json.loads(last)
    if {"ticker", "start_date", "end_date", "granularity"}.issubset(params):
        st.success(f"Parameters collected: {json.dumps(params)}")
        # TODO: proceed with data fetching using params
except json.JSONDecodeError:
    pass