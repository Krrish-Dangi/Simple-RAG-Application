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
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


## Pass Bytes of file to avoid cache miss
@st.cache_resource
def create_vector_store(file_bytes, size = 1000, overlap = 100, embedding_model = "qwen3-embedding:0.6b"):    
    ## Temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file_bytes)
        file_path = temp.name

    ## File Loader
    file_loader = PyPDFLoader(file_path)
    content = file_loader.load()

    ## Text Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = size, chunk_overlap = overlap)
    docs = text_splitter.split_documents(content)

    ## Embedding
    embedder = OllamaEmbeddings(model= embedding_model)

    ## Vector Store
    vector_store = Chroma.from_documents(documents=docs, embedding=embedder)

    return vector_store

@st.cache_resource
def get_llm(model_name = "llama2"):
    return OllamaLLM(model= model_name)

st.title("PDF-RAG")

st.subheader("Upload PDF and ask question from it...")

file_uploaded = st.file_uploader("Upload PDF",type="pdf")

if file_uploaded:

    with st.status("Creating Vector Store...", expanded=True) as status:
        ## create vector store DB
        vector_store = create_vector_store(file_bytes= file_uploaded.getvalue()) ## getvalue() returns bytes and buffer() returns memory view

        ## Vector Retriever
        status.update(
        label="✅ Vector Store Ready!",
        state="complete",
        expanded=False
        )

    vector_retreiver = vector_store.as_retriever()


question = st.text_input("Ask Question...")

if question and (not file_uploaded):
    st.error("Please Upload PDF..")

if question and file_uploaded:

    # ==========================================================
    # OLD MANUAL IMPLEMENTATION
    # ==========================================================

    # documents = ""
    # for i in vector_retreiver.invoke(question):
    #     documents = documents + i.page_content + "\n"

    # llm = OllamaLLM(model="llama2")

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant. Answer question asked with the help of given context:\n{documents}"),
    #     ("human", "{question}")
    # ])

    # output_parser = StrOutputParser()

    # chain = prompt | llm | output_parser

    # st.write(chain.invoke({
    #     "question": question,
    #     "documents": documents
    # }))

    # ==========================================================
    # NEW IMPLEMENTATION USING DOCUMENT CHAIN + RETRIEVAL CHAIN
    # ==========================================================

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer the user's question using the given context.\n\nContext:\n{context}"),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    retrieval_chain = create_retrieval_chain(
        vector_retreiver,
        document_chain
    )

    with st.status("🤖 Thinking...", expanded = True) as status:
        response = retrieval_chain.invoke({
            "input": question
        })

        status.update(
            label = "✅ Done", state = "complete", expanded = False
        )

    st.write(response["answer"])