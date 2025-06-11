import torch

model_id = "deepseek-ai/deepseek-llm-7b-chat"

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("mps")  
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device="mps")

# File: app.py

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
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

# LLM Pipeline Setup
@st.cache_resource
def load_llm_pipeline():
    model_id = "deepseek-ai/deepseek-llm-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("mps")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device="mps")
    return pipe

llm_pipeline = load_llm_pipeline()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_with_metadata(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                texts.append({"text": text, "metadata": {"source": pdf.name, "page": page_num}})
    return texts

def extract_text_with_metadata_from_csvs(csv_docs):
    texts = []
    for csv in csv_docs:
        df = pd.read_csv(csv, sep=';')
        for idx in range(len(df)):
            row = df.iloc[idx]
            row_text = ', '.join(f"{col}: {val}" for col, val in row.items())
            texts.append({"text": row_text, "metadata": {"source": csv.name, "row": idx}})
    return texts

def split_text_into_chunks(texts):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks_with_metadata = []
    for item in texts:
        chunks = text_splitter.split_text(item["text"])
        for chunk in chunks:
            chunks_with_metadata.append({"text": chunk, "metadata": item["metadata"]})
    return chunks_with_metadata

def build_vectorstore(text_chunks, existing_vectorstore=None):
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [chunk["metadata"] for chunk in text_chunks]
    embeddings = get_embeddings()
    if existing_vectorstore:
        existing_vectorstore.add_texts(texts=texts, metadatas=metadatas)
        return existing_vectorstore
    else:
        return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

def save_vectorstore(vectorstore, name):
    path = f"vectorstores/{name}"
    os.makedirs("vectorstores", exist_ok=True)
    vectorstore.save_local(path)
    st.success(f"Vectorstore saved as '{name}'.")

def load_vectorstore(name):
    path = f"vectorstores/{name}"
    embeddings = get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def answer_question_with_context(user_question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(user_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        f"Usa le informazioni qui sotto per rispondere alla domanda.\n"
        f"Se non trovi la risposta, dì che non lo sai.\n\n"
        f"Contesto:\n{context}\n\n"
        f"Domanda: {user_question}\nRisposta:"
    )
    response = llm_pipeline(prompt)[0]['generated_text']
    return response.split("Risposta:")[-1].strip(), docs

def handle_user_query(user_question):
    answer, docs = answer_question_with_context(user_question, st.session_state.vectorstore)
    st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)
    if docs:
        st.write("### Citazioni:")
        for doc in docs:
            meta = doc.metadata
            if 'page' in meta:
                info = f"Document: {meta['source']}, Page: {meta['page']}"
            elif 'row' in meta:
                info = f"CSV: {meta['source']}, Row: {meta['row']}"
            else:
                info = f"Fonte: {meta.get('source', 'n/d')}"
            st.write(f"- {info}")

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with PDFs and CSVs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_query(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        csv_docs = st.file_uploader("Upload CSVs", accept_multiple_files=True, type=['csv'])

        if "vectorstore" in st.session_state:
            operation = st.radio("Vectorstore operation", ("Create New", "Update Existing"))

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_texts = []
                if pdf_docs:
                    raw_texts.extend(extract_text_with_metadata(pdf_docs))
                if csv_docs:
                    raw_texts.extend(extract_text_with_metadata_from_csvs(csv_docs))
                text_chunks = split_text_into_chunks(raw_texts)
                if "vectorstore" in st.session_state and operation == "Update Existing":
                    vectorstore = build_vectorstore(text_chunks, st.session_state.vectorstore)
                else:
                    vectorstore = build_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore

        if "vectorstore" in st.session_state:
            save_name = st.text_input("Save vectorstore as")
            if st.button("Save Vectorstore"):
                if save_name:
                    save_vectorstore(st.session_state.vectorstore, save_name)

        load_name = st.text_input("Load vectorstore by name")
        if st.button("Load Vectorstore"):
            if load_name:
                try:
                    vectorstore = load_vectorstore(load_name)
                    st.session_state.vectorstore = vectorstore
                    st.success(f"Vectorstore '{load_name}' loaded successfully.")
                except Exception as e:
                    st.error(f"Error loading vectorstore: {e}")

main()
