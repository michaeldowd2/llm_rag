import numpy
import pandas as pd
import os
import json
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
from src.ragllm import ragllm

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
        ragllm(config_file, config, timestamp)