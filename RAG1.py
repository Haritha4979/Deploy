import os
import streamlit as st
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import time
from docx import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from google.api_core.exceptions import ResourceExhausted

# --- Patch asyncio to avoid 'no event loop' error --- #
nest_asyncio.apply()
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- Load Azure OpenAI credentials --- #
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY") or st.secrets.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL") or st.secrets.get("AZURE_OPENAI_MODEL", "gpt-4o-mini")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
    st.error("❌ Azure OpenAI credentials missing in .env or Streamlit Secrets.")
    st.stop()

# --- ✅ Add this function: Load DOCX from local path --- #
def load_docx_from_path(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# --- ✅ Keep only ONE split_text function --- #
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.create_documents([text])
    return [doc.page_content for doc in documents]

# --- Vectorstore from docs --- #
def create_vectorstore(texts):
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        deployment=AZURE_OPENAI_MODEL
    )
    return FAISS.from_texts(texts, embedding=embeddings)

# --- RAG Query --- #
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
    model = AzureChatOpenAI(
        openai_api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_MODEL,
        api_version="2024-02-15-preview"
    )

    try:
        response = model.invoke(prompt)
    except ResourceExhausted as e:
        if "quota" in str(e):
            time.sleep(11)
            response = model.invoke(prompt)
        else:
            raise e

    return response.content

# --- Streamlit UI --- #
st.set_page_config(page_title="📄 Chat with FAST Document", layout="centered")
st.title("📄 Chat with FAST_Workshop.docx (Azure OpenAI RAG)")

# --- Load document and create vectorstore only once --- #
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

# --- Chat Interface --- #
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
