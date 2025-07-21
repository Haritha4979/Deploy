import streamlit as st
import os
import asyncio

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Set your Google Gemini API key ===
GOOGLE_API_KEY = "AIzaSyDqhfgfkvTNkwj5RNMYKegm0bFEirr3IiQ"  # Replace with your actual key

# === Ensure valid API key ===
if not GOOGLE_API_KEY or "AIzaSy" not in GOOGLE_API_KEY:
    st.error("❌ Please set a valid Google Gemini API key.")
    st.stop()

# === Ensure event loop for asyncio ===
def ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            asyncio.set_event_loop(loop)
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

# === Async Gemini answer generator ===
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
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
    response = await model.ainvoke(prompt)
    return response.content

# === Async runner ===
def run_async(coro):
    ensure_event_loop()
    return asyncio.get_event_loop().run_until_complete(coro)

# === Load inbuilt document ===
def load_inbuilt_document():
    file_path = "FAST_Workshop.docx"  # Ensure this file is in the root folder
    suffix = os.path.splitext(file_path)[1]

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)
    elif suffix == ".txt":
        loader = TextLoader(file_path)
    else:
        st.error("❌ Unsupported file type.")
        return None

    return loader.load()

# === Streamlit App ===
st.title("📘 Ask Questions on Inbuilt Document")

if "vectordb" not in st.session_state:
    with st.spinner("📄 Loading and indexing document..."):
        try:
            documents = load_inbuilt_document()
            if not documents:
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            # Filter empty/whitespace chunks
            texts = [doc.page_content.strip() for doc in chunks if doc.page_content.strip()]
            metadatas = [doc.metadata for doc in chunks if doc.page_content.strip()]

            if not texts:
                st.error("❌ Document contains no valid content for embedding.")
                st.stop()

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

            vectordb = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
            st.session_state.vectordb = vectordb
            st.success("✅ Document indexed successfully!")

        except Exception as e:
            st.error(f"❌ Embedding or loading error: {e}")
            st.stop()

# === Chat UI ===
user_input = st.chat_input("💬 Ask a question about the document...")
if user_input and "vectordb" in st.session_state:
    st.chat_message("user").write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                answer = run_async(get_answer(st.session_state.vectordb, user_input))
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ Error answering question: {e}")
