# PDFSage
A Retrieval-Augmented Generation (RAG) chatbot that enables users to upload PDF documents and ask questions about their content. Built specifically for Hazard Identification and Risk Assessment workflows, but adaptable to any document-based Q&A use case.

## Features
- PDF Document Upload: Upload multiple PDF files through an intuitive web interface
- Intelligent Question Answering: Ask questions and get accurate answers extracted from your documents
- Source Citations: View the source document chunks used to generate each answer
- RAG Architecture: Combines retrieval and generation for accurate, context-aware responses
- User-Friendly Interface: Clean Streamlit-based UI for easy interaction

## Technology Stack
- Frontend: Streamlit
- LLM: OpenAI GPT-4
- Embeddings: OpenAI text-embedding-3-large
- Framework: LangChain
- Vector Store: InMemoryVectorStore
- Document Processing: PyPDF, RecursiveCharacterTextSplitter

  ## Dependencies
  Create a requirements.txt file with:
- streamlit
- langchain
- langchain-community
- langchain-openai
- langchain-text-splitters
= pypdf
- openai

  ##  Architecture
PDF Upload → Document Loading → Text Chunking → Embedding Generation → Vector Store -> User Query → Query Embedding → Similarity Search → Context Retrieval → GPT-4 Generation → Answer + Sources

## Key Components
**1. Document Processing**

- Loads PDFs using PyPDFDirectoryLoader
- Splits documents into 1000-character chunks with 200-character overlap
- Maintains context continuity across chunks

**2. Vector Store**

- Generates embeddings using OpenAI's text-embedding-3-large
- Stores embeddings in an in-memory vector database
- Enables fast similarity-based retrieval

**3. QA Chain**

- Uses RetrievalQA chain from LangChain
- Combines retrieved context with user query
- Generates answers using GPT-4 with temperature=0 for consistency

## Use Cases
- **Safety & Compliance:** Query HIRA documents, safety protocols, and compliance guidelines
- **Technical Documentation:** Search through manuals, specifications, and technical reports
- **Research:** Analyze research papers and extract relevant information
- **Legal Documents:** Query contracts, policies, and legal documents
