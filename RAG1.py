import os
import streamlit as st
from dotenv import load_dotenv
from docx import Document

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# Load environment
load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# UI
st.set_page_config(page_title="📄 Chat with FAST Workshop", layout="centered")
st.title("💬 Chatbot on FAST_Workshop.docx using Azure GPT-4o-mini")

@st.cache_resource
def load_and_embed_doc(path):
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.create_documents([text])

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment=embedding_model,
    )
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    return vectordb

try:
    file_path = os.path.join(os.path.dirname(__file__), "FAST_Workshop.docx")
    vectordb = load_and_embed_doc(file_path)

    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=chat_model,
    )

    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever())

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the FAST Workshop document...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa.run({"question": user_input, "chat_history": st.session_state.chat_history})
                st.markdown(result["answer"])
                st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})

except Exception as e:
    st.error(f"❌ Error: {e}")
