import os
import streamlit as st
from dotenv import load_dotenv
import asyncio
import nest_asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from docx import Document

nest_asyncio.apply()

try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- Load Gemini API Key --- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Google API Key missing in .env or Streamlit Secrets.")
    st.stop()

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
    return FAISS.from_documents(docs, embedding=embeddings)

# --- RAG Query --- #
def get_answer(vectorstore, query):
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
You are a helpful and knowledgeable assistant. Use the provided context to answer the user’s question accurately and clearly.

Instructions:
- Always base your answer strictly on the given context. Do not include external or fabricated information.
- Format your answer using **markdown**.
- Highlight important points using **bold**.
- Use bullet points or numbered steps if it improves clarity.
- Be concise but informative.
- If the context is insufficient to answer, state clearly: "The provided context does not contain enough information to answer the question."


Context:
{context}

Question:
{query}
"""
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
    response = model.invoke(prompt)
    return response.content

# --- Streamlit UI --- #
st.set_page_config(page_title="📄 Chat with FAST Document", layout="centered")
st.title("📄 Chat with FAST_Workshop.docx (Gemini RAG)")

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
                    answer = get_answer(st.session_state.vectordb, user_input)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")
