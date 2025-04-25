import os
import json
import datetime
import io
import streamlit as st
import mplfinance as mpf
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import wrds_ohlc_api as wrds
from dotenv import load_dotenv
from io import BytesIO


# --- Configuration ---
load_dotenv()
USRNAME = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
VALID_KEYS = {"ticker", "start_date", "end_date", "granularity"}

st.set_page_config(page_title="Smart WRDS", layout="wide")
st.title("📊 Stock Data Retriever")

# Load prompt
f = open("init_prompt.txt", "r", encoding="utf-8")
init_prompt = f.read()
f.close()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "type": "text", "content": init_prompt}]
if "params" not in st.session_state:
    st.session_state.params = None

# OpenAI Key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    OPENAI_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# RENDER chat history — now handles text/image/error inline
for msg in st.session_state.messages[1:]:  # skip system message
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "error":
            st.error(msg["content"])
        elif msg["type"] == "image":
            st.image(msg["content"], caption="📈 Candlestick Snapshot", use_container_width=True)

# New user input
if prompt := st.chat_input("Ask for stock data..."):
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not OPENAI_KEY:
        reply = "Please enter a valid OpenAI API Key in the sidebar."
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        # Run LLM
        lc_msgs = []
        for m in st.session_state.messages:
            if m["role"] == "system":
                lc_msgs.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_msgs.append(HumanMessage(content=m["content"]))
            elif m["type"] == "text":
                lc_msgs.append(AIMessage(content=m["content"]))

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        ai_out = llm.invoke(lc_msgs)

        try:
            parsed = json.loads(ai_out.content)
            if VALID_KEYS.issubset(parsed):
                st.session_state.params = parsed
            else:
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": ai_out.content})
                with st.chat_message("assistant"):
                    st.markdown(ai_out.content)
        except json.JSONDecodeError:
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": ai_out.content})
            with st.chat_message("assistant"):
                st.markdown(ai_out.content)

# Fetch + render inline chart if confirmed
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
                # raw_df = wrds.get_multi_month_data(db, ticker, start_date, end_date, granularity)
                # event_df = wrds.get_events(ticker)
                df = wrds.get_adjusted_ohlc(db, ticker, start_date, end_date, granularity)

                st.write("Do you want to download the data?")
                want_excel = st.radio("Do you want to download the data?", ("No", "Yes"))

                if want_excel == "Yes":
                    buffer = io.BytesIO()
                    df.to_excel(buffer, index=False)
                    buffer.seek(0)

                    
                    st.download_button(
                        label="Download OHLC data (Excel)",
                        data=buffer,
                        file_name="ohlc_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                
                df_mpf = df.set_index("timestamp")[["open", "high", "low", "close"]]
                df_mpf.index.name = "Date"

                buf = io.BytesIO()
                mpf.plot(df_mpf, type='candle', style='charles', ylabel='Price', savefig=buf)
                buf.seek(0)
                image = Image.open(buf)

                # Store result inline
                st.session_state.messages.append({"role": "assistant", "type": "image", "content": image})
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": "✅ Data successfully retrieved!"})
                st.image(image, caption="📈 Candlestick Snapshot", use_container_width=True)
                st.markdown("✅ Data successfully retrieved!")

        st.session_state.params = None

    except Exception as e:
        error_msg = f"❌ Error retrieving data: {e}"
        st.session_state.messages.append({"role": "assistant", "type": "error", "content": error_msg})
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.params = None
