import os
import json
import datetime
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import wrds_ohlc_api as wrds

# --- Configuration ---
USRNAME = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
VALID_KEYS = {"ticker", "start_date", "end_date", "granularity"}

st.set_page_config(page_title="Smart WRDS", layout="wide")
st.title("📊 Stock Data Retriever")

# Load system prompt from file
f = open("init_prompt.txt", "r")
init_prompt = f.read()
f.close()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": init_prompt}
    ]
if "params" not in st.session_state:
    st.session_state.params = None

# Get OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Render all previous chat messages
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask for stock data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not OPENAI_KEY:
        reply = "Please enter a valid OpenAI API Key in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
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

        try:
            parsed = json.loads(ai_out.content)
            if VALID_KEYS.issubset(parsed):
                st.session_state.params = parsed

            else:
                st.session_state.messages.append({"role": "assistant", "content": ai_out.content})
                with st.chat_message("assistant"):
                    st.markdown(ai_out.content)
        except json.JSONDecodeError:
            st.session_state.messages.append({"role": "assistant", "content": ai_out.content})
            with st.chat_message("assistant"):
                st.markdown(ai_out.content)

# --- Final Action: Fetch Data if Parameters Ready ---
if st.session_state.params:
    try:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving the data..."):
                params = st.session_state.params
                db = wrds.init_db_connection(USRNAME, PASSWORD)
                ticker = params["ticker"]
                start_date = datetime.datetime.strptime(params["start_date"], "%Y-%m-%d").date()
                end_date = datetime.datetime.strptime(params["end_date"], "%Y-%m-%d").date()
                granularity = params["granularity"]

                raw_df = wrds.get_multi_month_data(db, ticker, start_date, end_date, granularity)
                event_df = wrds.get_events(ticker)
                df = wrds.adjust_ohlc(raw_df, event_df)

            st.markdown("✅ Data successfully retrieved!")
            st.dataframe(df.head())

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"❌ Error retrieving data: {e}")
