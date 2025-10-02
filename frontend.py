import streamlit as st
from BackHIRA import (
    setup_environment,
    load_and_split_documents,
    save_uploaded_files,
    setup_vector_store,
    create_retrieval_qa_chain,
)

# Streamlit app setup
st.title("🔴 Hazard Identification and Risk Asssessment CHATBOT")
st.sidebar.header("Setup")

# API keys input
api_key_openai = st.sidebar.text_input("OpenAI API Key", type="password")

# File upload or directory input
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Files", type=["pdf"], accept_multiple_files=True
)
directory_path = st.sidebar.text_input("PDF Directory (Optional)", "Files_HIRA/")

# Save uploaded files and set directory path
if uploaded_files:
    directory_path = save_uploaded_files(uploaded_files)

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Process PDFs and set up QA chain
if st.sidebar.button("Process PDFs"):
    try:
        with st.spinner("Processing PDFs..."):
            setup_environment(api_key_openai)
            all_splits = load_and_split_documents(directory_path)
            vector_store = setup_vector_store(all_splits)
            st.session_state.qa_chain = create_retrieval_qa_chain(vector_store)
        st.success("PDFs processed and QA chain initialized!")
    except Exception as e:
        st.error(f"Error: {e}")

# Chat interface
st.header("Chat with your PDFs")
user_query = st.text_input("Enter your question:")

if st.button("Search"):
    if st.session_state.qa_chain:
        with st.spinner("Generating response..."):
            try:
                # Generate an answer using the QA chain
                result = st.session_state.qa_chain.invoke({"query": user_query})
                st.markdown("### Answer:")
                st.write(result["result"])
                st.markdown("### Sources:")
                for i, source in enumerate(result["source_documents"], 1):
                    st.markdown(f"**{i}.** {source.page_content[:200]}...")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Please process PDFs first.")
