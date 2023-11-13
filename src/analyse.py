import json
import os
import sys
import numpy as np
import time
import urllib.request
import traceback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup
import requests
import lxml
import logging

TEMPLATE_A = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. Answer in the form of a json list with minimal word length.
<context>
{context}
</context>
Question: {question}
Assistant:"""
TEMPLATE_B = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. Answer in the form of a json dictionary with the keys: "effect", "confidence" and "explanation".Don't include lists, apostrophes or quotes in any part of the answer.
<context>
{context}
</context>
Question: {question}
Assistant:"""
TEMPLATE_C = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. Answer in the form of a json dictionary with the keys: "change", "confidence" and "explanation".Don't include lists, apostrophes or quotes in any part of the answer.
<context>
{context}
</context>
Question: {question}
Assistant:"""

def Analyse(config_file, subject, no_factors, no_links):
    analysis_id = subject.replace(' ', '_')
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query = 'what are the current ' + str(no_factors) + ' biggest factors that effect the price of ' + subject
    SearchBingGetLinksAndSave(query, config_file, analysis_id, 'TL', no_links)
    CreateVectorDB(embeddings, config_file, analysis_id)

def RequestURLAndSave(url, config_file, analysis_id, doc_prefix, doc_id):
    try:
        headers = {
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582" 
        }
        logging.info('sending request for: ' + url)
        response = requests.get(url, headers = headers).text
        time.sleep(2)
        path = os.path.join(os.getcwd(),'output', config_file, 'html', analysis_id)
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info('created html output path at: ' + path)
        
        fname = os.path.join(path, doc_prefix + str(doc_id) + '.html')
        with open(fname, "w", encoding="utf-8") as f:
            f.write(str(response))
        logging.info('saved html file to: ' + fname)
        return response
    except Exception as e:
        logging.exception("error querying url or saving result")
        logging.exception(traceback.format_exc())
        raise Exception(e) 

def SearchBingGetLinksAndSave(query, config_file, analysis_id, doc_prefix, no_links):
    try:
        logging.info('running bing search for query: ' + query)
        url = "https://www.bing.com/search?form=QBRE&q="+query.replace(' ', '+')
        response = RequestURLAndSave(url, config_file, analysis_id, doc_prefix, 0)
        soup = BeautifulSoup(response, 'lxml')
        links = []
        for container in soup.select('.b_algo h2 a'):
            links.append(container['href'])
        links = links[:no_links]
        doc_id = 1
        for link in links:
            RequestURLAndSave(link, config_file, analysis_id, doc_prefix, doc_id)
            doc_id += 1
        logging.info('finished running and saving results for query')
    except Exception as e:
        logging.exception("error running bing search and save results")
        logging.exception(traceback.format_exc())
        raise Exception(e) 

def CreateVectorDB(embeddings, config_file, analysis_id):
    try:
        vector_db_path = os.path.join(os.getcwd(), 'output', config_file, 'html', analysis_id)
        vector_db_path += '\\'
        logging.info('loading vector db at directory: ' + vector_db_path)
        loader = DirectoryLoader(vector_db_path, loader_cls=BSHTMLLoader)
        documents = loader.load()
        logging.info('splitting documents')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        docs = text_splitter.split_documents(documents)
        logging.info('generating embeddings')
        vectorstore_faiss = FAISS.from_documents(docs, embeddings)
        return vectorstore_faiss
    except Exception as e:
        logging.exception("error generating embeddings")
        logging.exception(traceback.format_exc())
        raise Exception(e) 
    


