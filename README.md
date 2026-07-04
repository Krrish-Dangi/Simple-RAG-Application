# PDF-RAG

A simple Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain. The application allows users to upload a PDF and ask questions based on its contents.

## Features

* Upload a PDF document
* Automatic text chunking
* Embedding generation using Ollama
* Vector storage with ChromaDB
* Semantic retrieval of relevant document chunks
* Question answering using a local LLM
* Streamlit interface
* Cached vector store for improved performance

## Tech Stack

* Python
* Streamlit
* LangChain
* Ollama
* ChromaDB

## Project Workflow

1. Upload a PDF.
2. Load the document using `PyPDFLoader`.
3. Split the document into chunks using `RecursiveCharacterTextSplitter`.
4. Generate embeddings using Ollama Embeddings.
5. Store the embeddings in ChromaDB.
6. Retrieve the most relevant chunks for a user query.
7. Provide the retrieved context to the LLM.
8. Generate and display the final answer.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Create a virtual environment and activate it:

```bash
python -m venv venv
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file and add the following variables:

```env
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project_name
```
*Only iff you want to track api calls through langsmith*

## Running the Application

Start the Ollama server and ensure the required models are available.

Run the Streamlit application:

```bash
streamlit run app.py
```

## Models Used

* Embedding Model: `qwen3-embedding:0.6b`
* LLM: `llama2`

## Future Improvements

* Streaming responses
* Chat history
* Persistent vector database
* Support for multiple PDF uploads
* Source citations
* Configurable retrieval parameters

# *This project is intended for learning and educational purposes.*
