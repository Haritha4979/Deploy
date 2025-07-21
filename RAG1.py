import os
import streamlit as st
import asyncio
import nest_asyncio
import requests
from dotenv import load_dotenv
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama

# --- Async patch
nest_asyncio.apply()
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- Load environment variables
load_dotenv()

# --- Check if Ollama is running
def is_ollama_running():
    try:
        res = requests.get("http://localhost:11434")
        return res.status_code in [200, 404]
    except requests.exceptions.ConnectionError:
        return False

# --- Streamlit setup
st.set_page_config(page_title="📄 Chat with FAST Document (LLaMA)", layout="centered")
st.title("📄 Chat with FAST_Workshop.docx (LLaMA RAG)")

# --- Ollama server check
if not is_ollama_running():
    st.error("⚠️ Ollama server is not running. Please start it using `ollama serve`.")
    st.stop()

# --- Load DOCX
def load_docx_from_path(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# --- Split into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]

# --- Vectorstore using Ollama embeddings
def create_vectorstore(texts):
    embeddings = OllamaEmbeddings(model="llama3")
    return FAISS.from_texts(texts, embedding=embeddings)

# --- RAG QA with Ollama
def get_answer(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
You are a helpful assistant. Use the context below to answer the user's question.

Instructions:
- Write clearly in markdown.
- Use **bold** for key points and bullets or numbers where useful.
- Keep answers concise and grounded in the document.

Context:
{context}

Question:
{query}
"""
    llm = Ollama(model="llama3")
    return llm.invoke(prompt)

# --- Load and process DOCX
if "vectordb" not in st.session_state:
    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(file_dir, "FAST_Workshop.docx")

        text = load_docx_from_path(filepath)
        if not text.strip():
            st.warning("No extractable text found in document.")
            st.stop()

        st.success("✅ Document loaded successfully.")
        texts = split_text(text)
        st.session_state.vectordb = create_vectorstore(texts)
        st.session_state.chat_history = []

    except Exception as e:
        st.error(f"Error loading document: {e}")
        st.stop()

# --- Chat interface
if "vectordb" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about the FAST Workshop document...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Reading document..."):
                try:
                    answer = get_answer(st.session_state.vectordb, user_input)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
