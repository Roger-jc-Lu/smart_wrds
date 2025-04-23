import os
import json
import datetime
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import wrds_ohlc_api as wrds  # Your custom module

# Constants
USRNAME = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
VALID_KEYS = {"ticker", "start_date", "end_date", "granularity"}

# Streamlit page setup
st.set_page_config(page_title="Smart WRDS", layout="wide")
st.title("Adjusted Stock Data Retriever")

# OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Chatbot memory
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
   - Check that neither start_date nor end_date is relying on the real-time for example "today", "a month/week ago", etc, or goes beyond "2024-12-31".
    If so, set the end_date to "2024-12-31" and calculate the start_date respectively. Then ask for confirmation.

2. **Granularity fallback**  
   - If the user's granularity isn't in the allowed list (e.g. "3m"), compute the nearest lower and higher valid options (e.g. "2m" and "5m") and ask "Would you like 2m or 5m?".

3. **Ticker confirmation**  
   - If ticker is in lowercase, it is valid and normalize the ticker to uppercase.  
   - If ticker looks invalid, suggest a similar ticker and ask "Did you mean X? (yes/no)".

Continue asking follow-up questions until you have valid values for all four fields.  
Once collected, respond **only** with pure JSON, for example:
{"ticker":"AAPL","start_date":"2020-01-01","end_date":"2023-06-10","granularity":"1m"}
"""  
        }
    ]

# Param storage
if "params" not in st.session_state:
    st.session_state.params = None

# Input box
user_input = st.text_input("Enter your request", key="user_input")
if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not OPENAI_KEY:
        st.session_state.messages.append({"role": "assistant", "content": "Please enter a valid OpenAI API Key."})
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

        # Try parsing as JSON
        try:
            parsed = json.loads(ai_out.content)
            if VALID_KEYS.issubset(parsed):
                st.session_state.params = parsed
                st.session_state.messages.append({"role": "assistant", "content": "Retrieving the data..."})
            else:
                st.session_state.messages.append({"role": "assistant", "content": ai_out.content})
        except json.JSONDecodeError:
            st.session_state.messages.append({"role": "assistant", "content": ai_out.content})

# Render messages
for msg in st.session_state.messages[1:]:
    speaker = "**User**" if msg["role"] == "user" else "**Assistant**"
    st.markdown(f"{speaker}: {msg['content']}")

# Handle final output
if st.session_state.params:
    try:
        db = wrds.init_db_connection(USRNAME, PASSWORD)
        params = st.session_state.params
        ticker = params["ticker"]
        start_date = datetime.datetime.strptime(params["start_date"], "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(params["end_date"], "%Y-%m-%d").date()
        granularity = params["granularity"]

        df = wrds.get_adjusted_ohlc(db, ticker, start_date, end_date, granularity)
        st.success("✅ Data successfully retrieved!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
