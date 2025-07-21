import streamlit as st
import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Apply async patch for Streamlit compatibility
nest_asyncio.apply()

# === Load API Key from .env or Streamlit secrets ===
load_dotenv()
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# === Validate API key ===
if not GOOGLE_API_KEY or "AIzaSy" not in GOOGLE_API_KEY:
    st.error("❌ Please set a valid Google Gemini API key in .env or Streamlit secrets.")
    st.stop()

# === Streamlit UI Setup ===
st.set_page_config(page_title="FAST Gemini Chatbot", page_icon="🤖")
st.title("💬 FAST Workshop Chatbot")

# === Load and process document ===
def build_chatbot():
    loader = Docx2txtLoader("FAST_Workshop.docx")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    texts = [doc.page_content.strip() for doc in chunks if doc.page_content.strip()]
    metadatas = [doc.metadata for doc in chunks if doc.page_content.strip()]

    if not texts:
        st.error("❌ No valid content in the document to embed.")
        st.stop()

    def create_vectordb():
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        return FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    try:
        vectordb = create_vectordb()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        vectordb = loop.run_until_complete(asyncio.to_thread(create_vectordb))

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return qa_chain

# === Load chain once ===
if "qa_chain" not in st.session_state:
    with st.spinner("🔄 Loading and indexing document..."):
        st.session_state.qa_chain = build_chatbot()

# === Session state for conversation ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Chat Input ===
user_input = st.chat_input("💬 Ask something from the FAST Workshop document...")

if user_input:
    st.chat_message("user").markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.qa_chain({"question": user_input, "chat_history": st.session_state.chat_history})
                answer = result["answer"]
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("ai", answer))
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# === Display past messages ===
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
