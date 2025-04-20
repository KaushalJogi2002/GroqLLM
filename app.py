import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.title("📄 PDF Document Q&A using Gemma (LLaMA3)")

# Upload PDF
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

# Load LLM and prompt
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
""")

# Process PDFs and create vector embeddings
def process_uploaded_pdfs(files):
    docs = []
    for file in files:
        with open(f"temp_{file.name}", "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(f"temp_{file.name}")
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# Vector DB setup
if uploaded_files and st.button("🔍 Embed and Process PDFs"):
    with st.spinner("Processing and creating vector store..."):
        st.session_state.vectors = process_uploaded_pdfs(uploaded_files)
        st.success("Vector store is ready!")

# User question input
question = st.text_input("❓ Enter your question based on uploaded PDF documents")

# Answering logic
if question and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': question})
    st.write("🧠 **Answer:**", response['answer'])
    st.caption(f"⏱️ Response Time: {round(time.process_time() - start, 2)}s")

    with st.expander("🔍 Document Chunks Used"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.markdown("---")
elif question:
    st.warning("⚠️ Please upload and process PDFs first.")

