import numpy
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import sys

def run(config_file):
    print("loading config file: " + config_file)

if __name__ == "__main__":
    config_file = str(sys.argv[1])
    run(config_file)