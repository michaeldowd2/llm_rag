"""
Utilities for working with vector stores and embeddings.
Contains functions for creating and managing vector databases.
"""
import os
import logging
import traceback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from src.overrides.html_bs import BSHTMLLoader

def create_vector_db(config_file: str, analysis_id: str, prefix: str) -> FAISS:
    """
    Create a vector database from documents.
    
    Args:
        config_file: Configuration file name
        analysis_id: ID for the analysis
        prefix: Prefix for document loading
        
    Returns:
        FAISS: Vector store instance
    """
    try:
        # Set up vector database path
        vector_db_path = os.path.join(os.getcwd(), 'output', config_file, 'html', analysis_id)
        vector_db_path = os.path.join(vector_db_path, prefix) + '\\'
        logging.info('loading vector db at directory: ' + vector_db_path)
        
        # Load and process documents
        loader = DirectoryLoader(
            vector_db_path,
            loader_cls=BSHTMLLoader,
            loader_kwargs={'open_encoding': 'utf8', 'get_text_separator': '\n', 'get_text_strip': True}
        )
        documents = loader.load()
        
        # Split documents into chunks
        logging.info('splitting documents')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
        docs = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        logging.info('generating embeddings')
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        vectorstore_faiss = FAISS.from_documents(docs, embeddings)
        
        return vectorstore_faiss
        
    except Exception as e:
        logging.exception("error generating embeddings")
        logging.exception(traceback.format_exc())
        raise Exception(e)
