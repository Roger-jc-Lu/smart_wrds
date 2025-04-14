import streamlit as st
import os
import base64
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import PyPDF2

st.set_page_config(
    page_title="Simple RAG App",
    page_icon=":rocket:",
    layout="wide"
)

@st.cache_resource
def create_llm(api_key: str):
    """
    Cache the creation of the OpenAI LLM so it's not re-initialized on every run
    with the same key & temperature settings.
    """
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

@st.cache_resource
def create_vectorstore(_docs):
    """
    Cache the creation of the Chroma vector store. 
    If 'docs' has not changed, this won't be recomputed.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=_docs,
        embedding=embeddings,
        collection_name="financial_docs",
        persist_directory="./chroma_langchain_db",
    )

    return vectorstore

st.title("Upload Document RAG Application")
st.write("Upload your documents and ask questions about them.")

openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if not openai_api_key:
    st.warning("Please add your OpenAI API key to proceed.")
    st.stop()

llm = create_llm(openai_api_key)

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

all_docs = []


col1, col2 = st.columns([1,1])

with col1:
    st.subheader("PDF Previews")
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            pdf_data = uploaded_file.read()
            base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            pdf_display = (
                f'<iframe '
                f'src="data:application/pdf;base64,{base64_pdf}" '
                f'width="100%" height="500px" '
                f'type="application/pdf"></iframe>'
            )
            st.markdown(f"**File {i+1}:** {uploaded_file.name}")
            st.markdown(pdf_display, unsafe_allow_html=True)
            uploaded_files[i].seek(0)

with col2:
    st.subheader("Chat & Retrieval")
    if uploaded_files:
        all_texts = []
        for uploaded_file in uploaded_files:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join(
                page.extract_text() or ""
                for page in pdf_reader.pages
            )
            all_texts.append(text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        for txt in all_texts:
            chunks = text_splitter.split_text(txt)
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata={})
                all_docs.append(doc)

        if all_docs:
            vectorstore = create_vectorstore(all_docs)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            for message in st.session_state["messages"]:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])

            user_query = st.chat_input("Ask a question about your document(s) ...")
            if user_query:
                st.session_state["messages"].append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.write(user_query)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                    verbose=True
                )

                with st.spinner("Assistant is thinking..."):
                    result = qa_chain({"query": user_query})
                    answer = result["result"]
                    sources = result["source_documents"]

                st.session_state["messages"].append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.write(answer)
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(sources):
                            st.write(f"**Source {i+1}:** {doc.page_content[:500]}...")

    else:
        st.info("No PDF uploaded yet. Please upload one or more PDFs.")


st.header("Agent Demo: Calculator Tool")
st.write("Demonstration of using an agent with tools for simple tasks.")

from langchain.callbacks.base import BaseCallbackHandler

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, steps_placeholder):
        self.steps_placeholder = steps_placeholder
        self.step_logs = ""

    def on_tool_start(self, tool_name, tool_input, **kwargs):
        """Logs when a tool starts."""
        self.step_logs += f"**Tool Used:** {tool_name}\n**Input:** {tool_input}\n\n"
        self.steps_placeholder.markdown(self.step_logs, unsafe_allow_html=True)

    def on_tool_end(self, output, **kwargs):
        """Logs when a tool finishes."""
        self.step_logs += f"**Output:** {output}\n\n---\n\n"
        self.steps_placeholder.markdown(self.step_logs, unsafe_allow_html=True)

def calculator_tool(input_str: str) -> str:
    """Perform simple mathematical calculations."""
    try:
        return str(eval(input_str))
    except Exception as e:
        return f"Error: {str(e)}"

calc_tool = Tool(
    name="Calculator",
    func=calculator_tool,
    description="Perform simple mathematical calculations."
)

agent_tools = [calc_tool]
agent = initialize_agent(agent_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

agent_query = st.text_input("Ask the agent with tools (e.g., '2+2', '10*5', etc.)")
if agent_query:
    st.info("Agent is processing...")

    with st.expander("Step-by-Step Agent Logs", expanded=True):
        st.write("Processing steps will appear here...")
        steps_placeholder = st.empty()

    callback_handler = StreamlitCallbackHandler(steps_placeholder)

    try:
        response = agent.run(
            input=agent_query,
            callbacks=[callback_handler]
        )

        st.success("Agent processing complete!")
        st.write("**Agent Response:**", response)
    except Exception as e:
        st.error(f"Error: {str(e)}")
