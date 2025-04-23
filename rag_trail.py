import streamlit as st
import pandas as pd
import io
import os
from dotenv import load_dotenv # Import dotenv
from langchain_openai import ChatOpenAI # Updated import
from langchain_openai import OpenAIEmbeddings # Updated import
from langchain_community.vectorstores import FAISS # Updated import for community vectorstore
from langchain_community.document_loaders import CSVLoader # Updated import for community loader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback # Updated import for community callback
import traceback # Import traceback for better error logging

# --- Environment Variable Loading ---
# Load environment variables from a .env file if it exists
load_dotenv()

# --- Configuration ---
st.set_page_config(page_title="CSV RAG Assistant", layout="wide")
st.title("📄 CSV RAG Assistant")
st.write("Upload a CSV file and ask questions about its content.")
st.info("Ensure you have a `.env` file in the same directory with your `OPENAI_API_KEY` set.")

# --- API Key Handling ---
# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("🚨 OpenAI API Key not found! Please ensure it's set in your .env file (e.g., OPENAI_API_KEY='your_key_here').")
    st.stop() # Stop execution if no key is found

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# --- Global variable for the RAG chain ---
# We use st.session_state to keep the chain loaded across reruns
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'vector_store_loaded' not in st.session_state:
    st.session_state.vector_store_loaded = False
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

# --- RAG Chain Setup Function ---
# Use caching to avoid reloading and re-indexing the same file
@st.cache_resource(show_spinner="Processing CSV and building index...")
def setup_rag_chain(_uploaded_file, _api_key):
    """Loads CSV, creates embeddings, builds vector store, and sets up RAG chain."""
    # API key check is now done outside this function before calling

    # Define file_path and temp_dir initially to ensure they exist in finally block scope
    file_path = None
    temp_dir = "temp_csv_data"

    try:
        # Save uploaded file temporarily to disk because CSVLoader needs a file path
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        # Use the uploaded file's name for the temporary file
        file_path = os.path.join(temp_dir, _uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(_uploaded_file.getbuffer())
        st.write(f"Temporarily saved file to: {file_path}")

        # 1. Load CSV data
        # REMOVED source_column argument. CSVLoader will now automatically add row number
        # to metadata under the key 'row' (0-based index).
        loader = CSVLoader(file_path=file_path, encoding="utf-8") # No source_column specified
        documents = loader.load()
        st.write(f"Loaded {len(documents)} rows from the CSV.")

        if not documents:
            st.warning("CSV file loaded, but no documents were extracted. Is the CSV empty or formatted correctly?")
            # Clean up before returning
            if file_path and os.path.exists(file_path): os.remove(file_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir): os.rmdir(temp_dir)
            return None, False

        # 2. Create Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=_api_key)

        # 3. Create Vector Store (FAISS)
        # This creates embeddings for each document and stores them
        vector_store = FAISS.from_documents(documents, embeddings)
        st.write("Created vector store index (FAISS).")

        # 4. Create Retriever
        retriever = vector_store.as_retriever(search_kwargs={'k': 5}) # Retrieve top 5 relevant docs

        # 5. Create Prompt Template (Optional but recommended)
        template = """Use the following pieces of context from the CSV file to answer the question at the end.
        If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
        Keep the answer concise and directly based on the context provided.
        When referring to specific rows, mention the row number if available in the metadata.

        Context:
        {context}

        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # 6. Create RAG Chain (RetrievalQA)
        # Use a Chat model for better conversational ability
        llm = ChatOpenAI(openai_api_key=_api_key, model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Options: stuff, map_reduce, refine, map_rerank
            retriever=retriever,
            return_source_documents=True, # Return the source documents used
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        st.success("RAG chain setup complete!")
        return qa_chain, True

    except Exception as e:
        st.error(f"An error occurred during RAG setup: {e}")
        st.error(traceback.format_exc()) # Print full traceback for debugging
        return None, False

    finally:
        # Clean up temporary file and directory in all cases (success or failure)
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                st.write(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                st.warning(f"Could not remove temporary file {file_path}: {e}")
        if os.path.exists(temp_dir):
            try:
                # Only remove dir if it's empty
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                    st.write(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                # It's okay if removing the directory fails (e.g., permission error, or not empty unexpectedly)
                 st.warning(f"Could not remove temporary directory {temp_dir}: {e}")


# --- Main Logic ---
if uploaded_file is not None:
    # Check if it's a new file or the same one
    current_filename = uploaded_file.name
    if current_filename != st.session_state.get('last_uploaded_filename'): # Use .get for safety
        st.info(f"New file detected: {current_filename}. Resetting RAG chain.")
        st.session_state.vector_store_loaded = False # Reset flag for new file
        st.session_state.rag_chain = None
        st.session_state.last_uploaded_filename = current_filename
        # Clear the cache for the setup function when the file changes
        # Streamlit's @st.cache_resource handles this based on function arguments changing (_uploaded_file)

    # Only setup the chain if it hasn't been loaded for this file yet
    if not st.session_state.get('vector_store_loaded'): # Use .get for safety
        st.session_state.rag_chain, st.session_state.vector_store_loaded = setup_rag_chain(uploaded_file, openai_api_key)

    if st.session_state.get('rag_chain'): # Use .get for safety
        st.markdown("---")
        query = st.text_input("Ask a question about the CSV content:", key="query_input")

        if st.button("Get Answer", key="get_answer_button"):
            if query:
                with st.spinner("Searching data and generating answer..."):
                    try:
                        # Use the RAG chain to get the answer
                        # Wrap with callback to track token usage
                        with get_openai_callback() as cb:
                            result = st.session_state.rag_chain({"query": query})

                        st.subheader("Answer:")
                        st.markdown(result["result"]) # Use markdown for better formatting

                        st.subheader("Source Documents Used:")
                        # Display the relevant chunks retrieved from the CSV
                        if result.get("source_documents"):
                            for doc in result["source_documents"]:
                                st.write("---")
                                st.write(f"**Content:**")
                                st.text(doc.page_content) # Use st.text for preformatted view
                                # Display metadata like row number if available
                                # Check for the default 'row' key added by CSVLoader
                                if 'row' in doc.metadata:
                                     # Add 1 for human-readable row number (CSVLoader uses 0-based)
                                     # Add 1 more if there's a header row (usually the case)
                                     header_offset = 1 # Assume header row exists
                                     st.write(f"**Source Row:** {int(doc.metadata['row']) + 1 + header_offset}")
                                else:
                                     st.write(f"**Source Metadata:** {doc.metadata}") # Fallback
                        else:
                             st.write("No specific source documents were identified by the retriever for this query.")


                        st.subheader("Usage Info:")
                        st.write(f"Total Tokens: {cb.total_tokens}")
                        st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                        st.write(f"Completion Tokens: {cb.completion_tokens}")
                        st.write(f"Total Cost (USD): ${cb.total_cost:.6f}")


                    except Exception as e:
                        st.error(f"An error occurred while processing the query: {e}")
                        st.error(traceback.format_exc()) # Print full traceback for debugging
            else:
                st.warning("Please enter a question.")

else:
    # Reset state if no file is uploaded or file is removed
    if st.session_state.get('last_uploaded_filename') is not None:
         st.info("No file uploaded. Resetting state.")
         st.session_state.vector_store_loaded = False
         st.session_state.rag_chain = None
         st.session_state.last_uploaded_filename = None
    st.info("Please upload a CSV file to begin.")

st.markdown("---")
st.caption("Powered by LangChain, OpenAI, FAISS, and Streamlit")
