"""
Utilities for working with LLM models and processing their outputs.
Contains functions for loading models and parsing responses.
"""
import os
import json
import logging
import time
from typing import Dict, Any, Tuple, Optional
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def load_model(model_props: Dict[str, Any], grammar: str = '') -> LlamaCpp:
    """
    Load a LLaMA model with specified properties.
    
    Args:
        model_props: Dictionary of model properties
        grammar: Optional grammar file to use
        
    Returns:
        LlamaCpp: Loaded model instance
    """
    model_path = os.path.join(os.getcwd(), 'models', model_props['name'])
    model_args = {
        'model_path': model_path,
        'temperature': 0.0,
        'top_p': 1,
        'n_ctx': model_props['n_ctx'],
        'seed': 42,
        'verbose': True,
        'n_gpu_layers': model_props['n_gpu_layers'],
        'n_batch': model_props['n_batch']
    }
    
    if grammar:
        grammar_path = os.path.join(os.getcwd(), 'models', grammar)
        model_args['grammar_path'] = grammar_path
        
    return LlamaCpp(**model_args)

def load_context_and_run_llm(query: str, template: str, model_props: Dict[str, Any], 
                           grammar: str, vectorstore_faiss: Any, config_file: str, 
                           model: Optional[LlamaCpp] = None) -> Dict[str, Any]:
    """
    Load context and run LLM query.
    
    Args:
        query: Query string
        template: Prompt template
        model_props: Model properties
        grammar: Grammar file name
        vectorstore_faiss: FAISS vector store
        config_file: Config file name
        model: Optional pre-loaded model
        
    Returns:
        Dict[str, Any]: LLM response
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    if model is None:
        model = load_model(model_props, grammar)
        
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type='stuff',
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": model_props['k_docs']}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    logging.info('running llm query: ' + query)
    res = qa({"query": query})
    logging.info('llm query result: ' + str(res['result']))
    return res

def parse_json_result(response: Dict[str, Any], direction: str) -> Dict[str, Any]:
    """
    Parse JSON result from LLM response.
    
    Args:
        response: Raw LLM response
        direction: Direction context
        
    Returns:
        Dict[str, Any]: Parsed result
    """
    try:
        result = response['result']
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.endswith("```"):
            result = result[:-3]
            
        return json.loads(result)
        
    except Exception as e:
        logging.exception("error parsing json result")
        return {"answer": "", "explanation": str(e)}
