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
import inspect
import src.queries as queries

def Analyse(config_file, config, timestamp):

    subjects = config['subjects']
    directions = config['directions']
    no_factors= config['no_factors']
    no_links = config['no_links']
    model_props = config['model']
    
    factor_store = config['queries']['factor_store']
    factors_analysis = config['queries']['factors_analysis']
    effect_store = config['queries']['effect_store']
    effect_analyses = config['queries']['effect_analysis']
    func_dict = dict(inspect.getmembers(queries, inspect.isfunction))
    
    res = {}
    for subject in config['subjects']:
        res[subject] = {} # create dictionary for results
        SaveRes(res, config_file, timestamp)
        
        analysis_id = subject.replace(' ', '_')
        for direction in directions:
            res[subject][direction] = {} # create dictionary for results
            SaveRes(res, config_file, timestamp)
            
            prefix = direction + '_search'
            # create vector store with bing search results for top n factors
            vs = QueryBingAndCreateVectorStore(func_dict[factor_store], subject, '', direction, no_links, config_file, analysis_id, prefix)
            # use llm query to extract the top n factors
            factors = FindFactors(subject, func_dict[factors_analysis], direction, no_factors, no_links, vs, model_props, config_file, analysis_id)
            logging.info(factors)
            f_no = 0
            for factor in factors:
                if isNotANumber(factor): # sometimes llama returns a straight up number in the above list, skip this
                    res[subject][direction][factor] = {} # create dictionary for results
                    SaveRes(res, config_file, timestamp) 

                    prefix = direction + '_analysis_' + str(f_no)
                    # create vector store with one or more bing search queries
                    vs = QueryBingAndCreateVectorStore(func_dict[effect_store], subject, factor, direction, no_links, config_file, analysis_id, prefix)
                    for effect_analysis in effect_analyses: #analyse vector store with multiple llm queries
                        # analyse factor using llm prompt - basically try and find if the factor is likely to result in a price change
                        effect = AnalyseFactor(subject, factor, func_dict[effect_analysis], direction, no_links, vs, model_props, config_file, analysis_id, prefix)
                        res[subject][direction][factor][effect_analysis] = effect
                        SaveRes(res, config_file, timestamp)
                    f_no += 1

def FindFactors(subject, query, direction, no_factors, no_links, vectorstore_faiss, model_props, config_file, analysis_id):
    q, t = query()
    q = Parameterise(q, subject, '', no_factors, direction) 
    answer = LoadContextAndRunLLM(q, t, model_props, 'json_arr.gbnf', vectorstore_faiss, config_file, analysis_id, direction + '_factors')
    return ParseJSONResult(answer)

def AnalyseFactor(subject, factor, query, direction, no_links, vectorstore_faiss, model_props, config_file, analysis_id, prefix):
    q, t = query()
    q = Parameterise(q, subject, factor, '', direction)
    answer = LoadContextAndRunLLM(q, t, model_props, 'json.gbnf', vectorstore_faiss, config_file, analysis_id, prefix)
    return answer['result']

def QueryBingAndCreateVectorStore(query, subject, factor, direction, no_links, config_file, analysis_id, prefix):
    bs = query()
    bs = (Parameterise(b, subject, factor, no_links, direction) for b in bs)
    SearchBingGetLinksAndSave(bs, no_links, config_file, analysis_id, prefix)
    vs =  CreateVectorDB(config_file, analysis_id, prefix)
    return vs

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
    
def LoadContextAndRunLLM(query, template, model_props, grammar, vectorstore_faiss, config_file, analysis_id, prefix):
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
    return res

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
                time.sleep(0.143)
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

def Parameterise(inp, subject = '', factor = '', no_factors = '', direction = ''):
    inp = inp.replace('{subject}', subject)
    inp = inp.replace('{factor}', factor)
    inp = inp.replace('{no_factors}', str(no_factors))
    inp = inp.replace('{direction}', direction)
    return inp

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

def SaveRes(res, config_file, timestamp):
    path = os.path.join(os.getcwd(),'output', config_file, 'data')
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info('created data output path at: ' + path)
    fname = os.path.join(path, str(timestamp) + '.json')
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
        logging.info('saved result file to: ' + fname)



