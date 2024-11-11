import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from huggingface_hub import login
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

# HTML Templates
css = """
<style>
    .bot { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
    .user { background-color: #dfefff; padding: 10px; border-radius: 5px; }
</style>
"""
bot_template = '<div class="bot">{{MSG}}</div>'
user_template = '<div class="user">{{MSG}}</div>'

def extract_text_with_metadata(pdf_docs):
    """Extract text from PDFs and include metadata (source and page)."""
    texts = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                texts.append({"text": text, "metadata": {"source": pdf.name, "page": page_num}})
    return texts

def split_text_into_chunks(texts):
    """Split text into chunks with overlap and preserve metadata."""
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks_with_metadata = []
    for item in texts:
        chunks = text_splitter.split_text(item["text"])
        for chunk in chunks:
            chunks_with_metadata.append({"text": chunk, "metadata": item["metadata"]})
    return chunks_with_metadata

def build_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [chunk["metadata"] for chunk in text_chunks]
    return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

def handle_user_query(user_question, vectorstore, qa_pipeline):
    docs = vectorstore.similarity_search(user_question, k=5)
    context = " ".join([doc.page_content for doc in docs])
    
    result = qa_pipeline(question=user_question, context=context)
    answer = result["answer"]
    
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
    st.write("### Citations:")
    for doc in docs:
        st.write(f"- Document: {doc.metadata['source']}, Page: {doc.metadata['page']}")

def main():
    # Autenticazione Hugging Face con il token
    huggingface_token = st.secrets["HF_TOKEN"]
    login(token=huggingface_token)

    st.set_page_config(page_title="Chat with PDFs (Hugging Face)", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.vectorstore:
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
        handle_user_query(user_question, st.session_state.vectorstore, qa_pipeline)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_texts = extract_text_with_metadata(pdf_docs)
                text_chunks = split_text_into_chunks(raw_texts)
                st.session_state.vectorstore = build_vectorstore(text_chunks)

main()

