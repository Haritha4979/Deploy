
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from docx import Document
import pandas as pd
import PyPDF2

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---- Loaders for different file types ---- #
def load_txt(file):
    return file.read().decode("utf-8")

def load_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def load_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def load_xlsx(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)

# Splitting text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# Create vectorstore 
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    return FAISS.from_documents(docs, embeddings)

#  RAG Q&A  
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
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY
    )
    response = model.invoke(prompt)
    return response.content

#  Streamlit Chat UI
st.set_page_config(page_title="Chat with Your File", layout="centered")
st.title(" Chat with Your Document")

#  Upload and process file
if "vectordb" not in st.session_state:
    uploaded_file = st.file_uploader(
        "Upload a file (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file:
        try:
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "pdf":
                text = load_pdf(uploaded_file)
            elif ext == "docx":
                text = load_docx(uploaded_file)
            elif ext == "txt":
                text = load_txt(uploaded_file)
            else:
                st.error("Unsupported file format.")
                st.stop()

            if not text.strip():
                st.warning("No extractable text found.")
                st.stop()

            st.success(" File loaded successfully.")
            docs = split_text(text)
            st.session_state.vectordb = create_vectorstore(docs)
            st.session_state.chat_history = []

            #  Force rerun to show chat interface
            st.rerun()

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

#  Chat interface
if "vectordb" in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the uploaded document...")
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