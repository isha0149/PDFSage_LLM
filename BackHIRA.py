import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


# Set up API keys
def setup_environment(api_key_openai):
    os.environ["OPENAI_API_KEY"] = api_key_openai


# Load and process PDF files from a directory
def load_and_split_documents(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")
    
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No PDFs found in the directory: {directory_path}")
    
    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    
    return all_splits


# Save uploaded files into a specified directory
def save_uploaded_files(uploaded_files, save_directory="uploaded_pdfs"):
    os.makedirs(save_directory, exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join(save_directory, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
    return save_directory


# Embed documents and set up a vector store
def setup_vector_store(all_splits):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)
    return vector_store


# Create the RetrievalQA Chain
def create_retrieval_qa_chain(vector_store):
    """
    Create a chain for question answering using retrieval and LLM generation.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True  # Include the source documents in the output
    )
    return qa_chain
