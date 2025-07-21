import os
import streamlit as st
from dotenv import load_dotenv
from docx import Document
import nest_asyncio
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- Fix asyncio for Streamlit
nest_asyncio.apply()
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ Google API Key missing in .env or Streamlit secrets.")
    st.stop()

# --- Load built-in docx
def load_docx():
    filepath = os.path.join(os.path.dirname(__file__), "FAST_Workshop.docx")
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

# --- Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# --- Create FAISS vector store
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return FAISS.from_documents(docs, embedding=embeddings)

# --- Get answer from Gemini
def get_answer(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an expert assistant that answers user questions using the provided context extracted from a document

Instructions:
- Respond in markdown.
- Format your response in **markdown** for clarity and readability.
- Use **bold** for key facts, terms, or phrases.
- Use bullet points or numbered steps if the answer involves a list, process, or comparison.
- If the context does **not** provide enough information to answer the question, say:
  > " The provided document does not contain enough information to answer this question."
- Be clear, concise, and structured 
Context:
{context}

Question:
{query}
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    return model.invoke(prompt).content

# --- Streamlit UI
st.set_page_config(page_title="📄 Gemini RAG Chatbot", layout="centered")
st.title("📄 Chat with FAST_Workshop.docx (Gemini RAG)")

if "vectordb" not in st.session_state:
    try:
        text = load_docx()
        if not text.strip():
            st.warning("Document is empty.")
            st.stop()
        docs = split_text(text)
        st.session_state.vectordb = create_vectorstore(docs)
        st.session_state.chat_history = []
        st.success("✅ Document loaded successfully.")
    except Exception as e:
        st.error(f"Error loading document: {e}")
        st.stop()

if "vectordb" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = get_answer(st.session_state.vectordb, user_input)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
