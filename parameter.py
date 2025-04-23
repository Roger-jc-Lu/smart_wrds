import os
import json
import datetime
import streamlit as st
from langchain_openai import ChatOpenAI
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

f = open("init_prompt.txt", "r")
init_prompt = f.read()
f.close()

# Chatbot memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": init_prompt
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

        raw_df = wrds.get_multi_month_data(db, ticker, start_date, end_date, granularity)
        event_df = wrds.get_events(db, ticker, start_date, end_date)
        # TODO: event came from either the above function call or RAG if symbol in RAG
        df = wrds.adjust_ohlc(raw_df, event_df)

        # CANT USE THIS, only for showcasing
        # df = wrds.get_adjusted_ohlc(db, ticker, start_date, end_date, granularity)
        st.success("✅ Data successfully retrieved!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
