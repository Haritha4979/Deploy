import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from docx import Document

# --- Load API key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Gemini API key missing. Add it to .env or Streamlit secrets.")
    st.stop()

# --- Load DOCX file directly ---
def load_docx():
    doc = Document("FAST_Workshop.docx")
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# --- Text splitter ---
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# --- Create vectorstore ---
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    return Chroma.from_documents(docs, embeddings, collection_name="doc_collection")

# --- RAG logic ---
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
        google_api_key=GEMINI_API_KEY
    )
    response = model.invoke(prompt)
    return response.content

# --- Streamlit UI ---
st.set_page_config(page_title="Chat Assistant", layout="centered")
st.title("Chat with Your Document")

# --- Load and prepare document ---
if "vectordb" not in st.session_state:
    try:
        text = load_docx("FAST Workshop.docx")
        if not text.strip():
            st.error("❌ No extractable text found in the document.")
            st.stop()

        docs = split_text(text)
        st.session_state.vectordb = create_vectorstore(docs)
        st.session_state.chat_history = []
        st.success("✅ Document loaded successfully.")
    except Exception as e:
        st.error(f"Error loading document: {e}")
        st.stop()

# --- Chat UI ---
if "vectordb" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the FAST Workshop...")
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
