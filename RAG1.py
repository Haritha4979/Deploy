import streamlit as st
import os
import asyncio
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile

# Set your Gemini API key here
GOOGLE_API_KEY = "your_google_api_key"

# === Safe async wrapper ===
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            return new_loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

# === Async answer generator ===
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

# === Document loader ===
def load_document(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    if suffix == ".pdf":
        loader = PyPDFLoader(temp_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(temp_path)
    elif suffix == ".txt":
        loader = TextLoader(temp_path)
    else:
        st.error("Unsupported file type.")
        return None

    return loader.load()

# === Main App ===
st.title("📄 Document Q&A with Gemini")

uploaded_file = st.file_uploader("Upload a PDF, Word or Text file", type=["pdf", "docx", "txt"])
if uploaded_file:
    if "vectordb" not in st.session_state:
        with st.spinner("Loading and indexing document..."):
            documents = load_document(uploaded_file)
            if documents:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(documents)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                vectordb = FAISS.from_documents(chunks, embedding=embeddings)
                st.session_state.vectordb = vectordb
                st.success("Document loaded successfully!")

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
