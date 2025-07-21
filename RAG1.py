import streamlit as st
import os
import asyncio

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# === Set Gemini API Key ===
GOOGLE_API_KEY = "your_google_api_key"  # <-- Replace with your key

# === Ensure event loop for async gRPC ===
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# === Async wrapper for Streamlit compatibility ===
def run_async(coro):
    ensure_event_loop()
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

# === Async Gemini Q&A ===
async def get_answer(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
You are a helpful assistant. Use the context below to answer the user's question.

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

# === Load local inbuilt document ===
def load_inbuilt_document():
    file_path = "FAST_Workshop.docx"  # 👈 Make sure this file exists here
    suffix = os.path.splitext(file_path)[1].lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)
    elif suffix == ".txt":
        loader = TextLoader(file_path)
    else:
        st.error("Unsupported file type.")
        return None

    return loader.load()

# === Streamlit App ===
st.title("📘 Ask Questions on Inbuilt Document")

# Initialize vector DB once
if "vectordb" not in st.session_state:
    with st.spinner("Loading and indexing inbuilt document..."):
        documents = load_inbuilt_document()
        if documents:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            ensure_event_loop()  # 🔄 Fix for gRPC
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

            vectordb = FAISS.from_documents(chunks, embedding=embeddings)
            st.session_state.vectordb = vectordb
            st.success("Document loaded and indexed successfully!")

# Chat UI
user_input = st.chat_input("Ask a question about the document...")
if user_input and "vectordb" in st.session_state:
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = run_async(get_answer(st.session_state.vectordb, user_input))
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
