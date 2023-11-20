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
from .queries import Factors_Q0
from .queries import Effect_Q0
from .queries import Effect_Q1
from .queries import Effect_Q2

def Analyse(config_file, config, timestamp):
    no_factors= config['no_factors']
    no_links = config['no_links']
    model_props = config['model']
    
    res = {}
    for subject in config['subjects']:
        res[subject] = {} # create dictionary for results
        analysis_id = subject.replace(' ', '_')
        for direction in ['increase', 'decrease']:
            res[subject][direction] = {} # create dictionary for results
            factors = FindFactors(subject, Factors_Q0, direction, no_factors, no_links, model_props, config_file, analysis_id)
            logging.info(factors)
            f_no = 0
            for factor in factors:
                if isNotANumber(factor):
                    res[subject][direction][factor] = {} # create dictionary for results
                    SaveRes(res, config_file, timestamp)
                    q_no, bing_dl = 0, True
                    for query in [Effect_Q0, Effect_Q1, Effect_Q2]: #loop through effect queries
                        if q_no > 0:
                            bing_dl = False
                        q_name = str(query.__name__)
                        prefix = direction + '_F' + str(f_no)
                        effect = AnalyseFactor(subject, factor, query, direction, no_links, model_props, bing_dl, config_file, analysis_id, prefix)
                        res[subject][direction][factor][q_name] = effect
                        SaveRes(res, config_file, timestamp)
                        q_no+=1
                    f_no += 1

def FindFactors(subject, query, direction, no_factors, no_links, model_props, config_file, analysis_id):
    bs, q, t = query()
    bs = (Parameterise(b, subject, '', no_factors, direction) for b in bs)
    q = Parameterise(q, subject, '', no_factors, direction) 
    SearchBingGetLinksAndSave(bs, no_links, config_file, analysis_id, direction + '_factors')
    answer = LoadContextAndRunLLM(q, t, model_props, 'json_arr.gbnf', config_file, analysis_id, direction + '_factors')
    print('factor result')
    print(answer)
    return ParseJSONResult(answer)

def AnalyseFactor(subject, factor, query, direction, no_links, model_props, bing_dl, config_file, analysis_id, prefix):
    bs, q, t = query()
    bs = (Parameterise(b, subject, factor, '', direction) for b in bs)
    q = Parameterise(q, subject, factor, '', direction) 
    if bing_dl:
        SearchBingGetLinksAndSave(bs, no_links, config_file, analysis_id, prefix)
    answer = LoadContextAndRunLLM(q, t, model_props, 'json.gbnf', config_file, analysis_id, prefix)
    return answer['result']

def Parameterise(inp, subject = '', factor = '', no_factors = '', direction = ''):
    inp = inp.replace('{subject}', subject)
    inp = inp.replace('{factor}', factor)
    inp = inp.replace('{no_factors}', str(no_factors))
    inp = inp.replace('{direction}', direction)
    return inp

def isNotANumber(factor):
    try:
        float(factor)
        return False # not a valid factor if it can be converted to a float
    except:
        pass
    try:
        int(factor)
        return False
    except:
        pass
    return True
    
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

def RequestURLAndSave(url, save, config_file, analysis_id, doc_prefix, doc_id):
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
        
        if save:
            fname = os.path.join(path, str(doc_id) + '.html')
            with open(fname, "w", encoding="utf-8") as f:
                f.write(str(response))
            logging.info('saved html file to: ' + fname)
        return response
    except Exception as e:
        logging.exception("error querying url or saving result")
        logging.exception(traceback.format_exc())
        raise Exception(e) 

def SearchBingGetLinksAndSave(queries, no_links, config_file, analysis_id, doc_prefix):
    try:
        doc_id = 0
        for query in queries:
            logging.info('running bing search for query: ' + query)
            url = "https://www.bing.com/search?form=QBRE&q="+query.replace(' ', '+')
            response = RequestURLAndSave(url, False, config_file, analysis_id, doc_prefix, doc_id)
            soup = BeautifulSoup(response, 'lxml')
            links = []
            for container in soup.select('.b_algo h2 a'):
                actURL = BingToActualURL(container['href']) # need to find the actual url
                if actURL is not None:
                    links.append(actURL)
            links = links[:no_links]
            for link in links:
                RequestURLAndSave(link, True, config_file, analysis_id, doc_prefix, doc_id)
                doc_id += 1
            logging.info('finished running and saving results for query')
    except Exception as e:
        logging.exception("error running bing search and save results")
        logging.exception(traceback.format_exc())
        raise Exception(e) 

def BingToActualURL(url):
    headers = {
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582" 
        }
    response = requests.get(url, headers = headers).text
    start_ind = response.index('var u = "')
    if start_ind >= 0:
        response = response[start_ind+9:]
        end_ind = response.find('";')
        if end_ind >=0:
            response = response[:end_ind]
            return response
    return None

def CreateVectorDB(config_file, analysis_id, prefix):
    try:
        vector_db_path = os.path.join(os.getcwd(), 'output', config_file, 'html', analysis_id)
        vector_db_path = os.path.join(vector_db_path, prefix) + '\\'
        logging.info('loading vector db at directory: ' + vector_db_path)
        loader = DirectoryLoader(vector_db_path, loader_cls=BSHTMLLoader, loader_kwargs={'open_encoding':'utf8'})
        documents = loader.load()
        logging.info('splitting documents')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
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
        return_source_documents = True,
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
