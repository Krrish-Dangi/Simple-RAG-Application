import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("PDF-RAG")

st.subheader("Upload PDF and ask question from it...")

file_uploaded = st.file_uploader("Upload PDF",type="pdf")

if file_uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file_uploaded.getbuffer())
        file_path = temp.name
    
    ## File Loader
    file_loader = PyPDFLoader(file_path)
    content = file_loader.load()

    ## Text Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = text_splitter.split_documents(content)

    ## Embedding
    embeder = OllamaEmbeddings(model="qwen3-embedding:0.6b")

    ## Vector Store
    vector_store = Chroma.from_documents(documents=docs, embedding=embeder)

    ## Vector Retreiver
    vector_retreiver = vector_store.as_retriever()


question = st.text_input("Ask Question...")

if question and (not file_uploaded):
    st.error("Please Upload PDF..")

if question and file_uploaded:
    documents = ""
    for i in vector_retreiver.invoke(question):
        documents = documents + i.page_content + "\n"

    llm = OllamaLLM(model="llama2")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer question asked with the help of given context:\n{documents}"),
        ("human", "{question}")
    ])

    output_parser = StrOutputParser()

    chain = prompt|llm|output_parser
    st.write(chain.invoke({"question": question, "documents": documents}))
    