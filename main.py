from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv


class DocumentQA:
    def __init__(self, docs_dir="./documents"):
        self.docs_dir = docs_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.llm = HuggingFaceHub(
            repo_id="meta-llama/Llama-3.2-1B",
            model_kwargs={
                "temperature": 0.3,  
                "max_length": 512,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        )

    def load_documents(self):
        documents = []
        
        txt_loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents.extend(txt_loader.load())
        
        pdf_loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents):
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        return vectorstore

    def setup_qa_chain(self, vectorstore):
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

    def answer_question(self, qa_chain, question):
        formatted_question = f"""Analyze the context carefully and provide a clear, concise answer to the following question:

Context Provided
Question: {question}

Your Response:"""
        
        response = qa_chain({"query": formatted_question})
        return {
            "answer": response["result"].strip(),
            "source_documents": [doc.page_content for doc in response["source_documents"]]
        }

def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return temp_dir

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Document Q&A System", layout="wide")
    st.title("\ud83d\udcc4 Intelligent Document Q&A System")
    
    st.sidebar.header("About")
    st.sidebar.info(
        "Upload PDF or TXT documents and ask questions. "
        "The system uses AI to understand and answer based on the context."
    )
    
    st.subheader("\ud83d\udcc4 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files (Multiple files supported)",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        temp_dir = save_uploaded_files(uploaded_files)
        
        with st.spinner("\ud83d\udd04 Initializing AI System..."):
            qa_system = DocumentQA(docs_dir=temp_dir)
            
            progress_bar = st.progress(0)
            st.text("\ud83d\udcc2 Loading documents...")
            documents = qa_system.load_documents()
            progress_bar.progress(33)
            
            st.text("\ud83d\uddc3\ufe0f Creating vector store...")
            vectorstore = qa_system.create_vector_store(documents)
            progress_bar.progress(66)
            
            st.text("\ud83e\udd16 Setting up AI chain...")
            qa_chain = qa_system.setup_qa_chain(vectorstore)
            progress_bar.progress(100)
            
            progress_bar.empty()
            st.success("\ud83d\ude80 System is ready!")
        
        st.subheader("❓ Ask Your Question")
        question = st.text_input("Enter your question about the uploaded documents:")
        
        if question:
            try:
                with st.spinner("\ud83e\udde0 Generating intelligent response..."):
                    response = qa_system.answer_question(qa_chain, question)
                
                st.subheader("\ud83d\udca1 Answer")
                st.write(response["answer"])
                
                with st.expander("\ud83d\udcc4 View Source Documents"):
                    for i, doc in enumerate(response["source_documents"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc)
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.warning("\ud83d\udd0d Please try rephrasing your question or check the uploaded documents.")
        
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
