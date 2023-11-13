import numpy
import pandas as pd
import os
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import sys
import logging
import datetime
from wakepy import keep
import json
from src.analyse import Analyse

def Run(config_file, config):
    for subject in config['subjects']:
        Analyse(config_file, subject, config["no_factors"], config["no_links"], )
    return
    llm = LoadModel(config)
    question = "provide details of the book Innovator's dilemma in a json dictionary with keys: writer, year and description."
    answer = llm(question)
    logging.info(answer)

def LoadModel(config):
    if 'model' not in config:
        logging.exception("model name not included in config")
    model_path = os.path.join(os.getcwd(), 'models', config['model'])
    if not os.path.exists(model_path):
        logging.exception("model_path not exist: " + model_path)
    try:
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.0,
            top_p=1,
            n_ctx=6000,
            verbose=True,
        )
    except Exception as e:
        logging.exception(traceback.format_exc())
        raise Exception(e)
    logging.info("loaded model: " + model_path)
    return llm

def CreateFolders(config_file):
    logging.info("Creating folder structure")
    if not os.path.exists(os.path.join(os.getcwd(),'output',config_file)):
        os.mkdir(os.path.join(os.getcwd(),'output',config_file))
    if not os.path.exists(os.path.join(os.getcwd(),'output',config_file,'data')):
        os.mkdir(os.path.join(os.getcwd(),'output',config_file,'data'))
    if not os.path.exists(os.path.join(os.getcwd(),'output',config_file,'html')):
        os.mkdir(os.path.join(os.getcwd(),'output',config_file,'html'))

def ReadConfigFile(config_file):
    file_path = os.path.join(os.getcwd(),'config',config_file+'.json')
    if not os.path.exists(file_path):
        logging.exception("config file does not exist")
    f = open(file_path)
    config = json.load(f)
    return config

if __name__ == "__main__":
    with keep.running() as k:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        config_file = str(sys.argv[1])
        
        log_file_name = os.path.join(os.getcwd(), 'logs', config_file + '_' + timestamp + '.log')
        logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode="w")
        
        CreateFolders(config_file)
        config = ReadConfigFile(config_file)
        
        Run(config_file, config)