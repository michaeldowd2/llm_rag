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

def run(config_file, timestamp):
    CreateConfigFolders(config_file)
    log_file_name = os.path.join(os.getcwd(),'output',config_file,'logs') + '/' + timestamp + '.log'
    logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode="w")
    logging.info("Starting Run for: " + config_file + ', at: ' + timestamp)

def CreateConfigFolders(config_file):
    if not os.path.exists(os.path.join(os.getcwd(),'output',config_file)):
        os.mkdir(os.path.join(os.getcwd(),'output',config_file))
    if not os.path.exists(os.path.join(os.getcwd(),'output',config_file,'data')):
        os.mkdir(os.path.join(os.getcwd(),'output',config_file,'data'))
    if not os.path.exists(os.path.join(os.getcwd(),'output',config_file,'html')):
        os.mkdir(os.path.join(os.getcwd(),'output',config_file,'html'))
    if not os.path.exists(os.path.join(os.getcwd(),'output',config_file,'logs')):
        os.mkdir(os.path.join(os.getcwd(),'output',config_file,'logs'))

if __name__ == "__main__":
    with keep.running() as k:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        config_file = str(sys.argv[1])
        run(config_file, timestamp)