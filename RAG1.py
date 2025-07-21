import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- Load Gemini API Key --- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Google API Key missing in .env or Streamlit Secrets.")
    st.stop()

# --- Fix for async calls inside Streamlit thread --- #
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside another loop (like Streamlit) — create a new loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)

# --- Load DOCX from local path --- #
def load_docx_from_path(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# --- Split text into chunks --- #
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# --- Vectorstore from docs --- #
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return Chroma.from_documents(docs, embeddings, collection_name="doc_collection")

# --- Gemini RAG Query --- #
async def get_answer(vectorstore, query):
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
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GOOGLE_API_KEY
    )
    response = await model.ainvoke(prompt)
    return response.content

# --- Streamlit UI --- #
st.set_page_config(page_title="📄 Chat with FAST Document", layout="centered")
st.title("📄 Chat with FAST_Workshop.docx (Gemini RAG)")

# --- Load document and create vectorstore only once --- #
if "vectordb" not in st.session_state:
    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(file_dir, "FAST_Workshop.docx")  # Ensure the file is in your project root

        text = load_docx_from_path(filepath)

        if not text.strip():
            st.warning("No extractable text found in document.")
            st.stop()

        st.success("✅ Document loaded successfully.")
        docs = split_text(text)
        st.session_state.vectordb = create_vectorstore(docs)
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
                    answer = run_async(get_answer(st.session_state.vectordb, user_input))
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
