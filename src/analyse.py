import json
import os
import sys
import numpy as np
import time
import urllib.request
import traceback
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
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
from .queries import BINGQ_Factors
from .queries import LLMQ_Factors
from .queries import BINGQ_Effect
from .queries import LLMQ_Effect

def Analyse(config_file, config, timestamp):
    no_factors= config['no_factors']
    no_links = config['no_links']
    model_props = config['model']
    
    res = {}
    for subject in config['subjects']:
        res[subject] = {'factors':{}} 
    
    for subject in config['subjects']:
        analysis_id = subject.replace(' ', '_')
        factors = FindFactors(subject, no_factors, no_links, model_props, config_file, analysis_id)
        logging.info('factors returned')
        logging.info(factors)
        for factor in factors:
            res[subject]['factors'][factor]={}
        SaveRes(res, config_file, timestamp)

        factorid = 0
        for factor in factors:
            factor_increase = AnalyseFactor(subject, factor, no_links, model_props, config_file, analysis_id, factorid)
            if (factor_increase != None):
                logging.info('factor: ' + str(factorid) + ' returned')
                logging.info(factor_increase)
                res[subject]['factors'][factor]['increase'] = factor_increase
                SaveRes(res, config_file, timestamp)
            factorid += 1

def FindFactors(subject, no_factors, no_links, model_props, config_file, analysis_id):
    bq = BINGQ_Factors(subject, no_factors)
    q, t = LLMQ_Factors(subject, no_factors)
    SearchBingGetLinksAndSave(bq, config_file, analysis_id, 'factors', no_links)
    answer = LoadContextAndRunLLM(q, t, model_props, 'json_arr.gbnf', config_file, analysis_id, 'factors')
    return ParseJSONResult(answer)

def AnalyseFactor(subject, factor, no_links, model_props, config_file, analysis_id, factorid):
    bq = BINGQ_Effect(subject, factor)
    q, t = LLMQ_Effect(subject, factor)
    SearchBingGetLinksAndSave(bq, config_file, analysis_id, 'factor'+str(factorid), no_links)
    answer = LoadContextAndRunLLM(q, t, model_props, 'json.gbnf', config_file, analysis_id, 'factor'+str(factorid))
    return ParseJSONResult(answer)

def LoadModel(model_props, grammar):
    model_path = os.path.join(os.getcwd(), 'models', model_props['name'])
    grammar_path = os.path.join(os.getcwd(), 'models', grammar)
    llm = LlamaCpp(model_path=model_path, 
        temperature=0.0, 
        top_p=1,
        n_ctx=model_props['n_ctx'], 
        seed = 42,
        verbose=True, 
        n_gpu_layers=model_props['n_gpu_layers'],
        n_batch=model_props['n_batch'],
        grammar_path=grammar_path
    )
    return llm

def SaveRes(res, config_file, timestamp):
    path = os.path.join(os.getcwd(),'output', config_file, 'data')
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info('created data output path at: ' + path)
    fname = os.path.join(path, str(timestamp) + '.json')
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
        logging.info('saved result file to: ' + fname)


def RequestURLAndSave(url, config_file, analysis_id, doc_prefix, doc_id):
    try:
        headers = {
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582" 
        }
        logging.info('sending request for: ' + url)
        response = requests.get(url, headers = headers).text
        time.sleep(2)
        path = os.path.join(os.getcwd(),'output', config_file, 'html', analysis_id)
        path = os.path.join(path, doc_prefix)
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info('created html output path at: ' + path)
        
        fname = os.path.join(path, str(doc_id) + '.html')
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

def CreateVectorDB(config_file, analysis_id, prefix):
    try:
        vector_db_path = os.path.join(os.getcwd(), 'output', config_file, 'html', analysis_id)
        vector_db_path = os.path.join(vector_db_path, prefix) + '\\'
        logging.info('loading vector db at directory: ' + vector_db_path)
        loader = DirectoryLoader(vector_db_path, loader_cls=BSHTMLLoader, loader_kwargs={'open_encoding':'utf8'})
        documents = loader.load()
        logging.info('splitting documents')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 750, chunk_overlap = 50)
        docs = text_splitter.split_documents(documents)
        logging.info('generating embeddings')
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        vectorstore_faiss = FAISS.from_documents(docs, embeddings)
        return vectorstore_faiss
    except Exception as e:
        logging.exception("error generating embeddings")
        logging.exception(traceback.format_exc())
        raise Exception(e) 
    
def LoadContextAndRunLLM(query, template, model_props, grammar, config_file, analysis_id, prefix):
    vectorstore_faiss = CreateVectorDB(config_file, analysis_id, prefix)
    prompt = PromptTemplate(template = template, input_variables = ["context", "question"])
    model = LoadModel(model_props, grammar)
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type='stuff',
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k":5}),
        return_source_documents = False,
        chain_type_kwargs = {"prompt":prompt}
    )
    res = qa({"query": query})
    model, vectorstore_faiss = None, None
    time.sleep(10)
    return res

def ParseJSONResult(answer):
    try:
        res = json.loads(answer['result'])
        return res
    except Exception as e:
        logging.exception('failed to parse json: ')
        logging.exception(answer['result'])
        logging.exception(e)
        return None
        raise Exception(e)
