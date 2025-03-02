"""
Main module for RAG-based LLM analysis of financial securities.
Coordinates the process of collecting data, running analysis, and generating signals.
"""
import json
import os
import time
import logging
import inspect
from typing import Dict, Any, List

from src.llm_utils import load_model, load_context_and_run_llm, parse_json_result
from src.web_utils import search_bing_get_links_and_save, request_url_and_save, bing_to_actual_url
from src.vector_utils import create_vector_db
import src.queries as queries

def ragllm(config_file: str, config: Dict[str, Any], timestamp: str) -> None:
    """
    Main function for running RAG-LLM analysis.
    
    Args:
        config_file: Name of the configuration file
        config: Configuration dictionary
        timestamp: Timestamp for the analysis run
    """
    # Extract configuration
    subjects = config['subjects']
    directions = config['directions']
    no_factors = config['no_factors']
    no_links = config['no_links']
    model_props = config['model']
    
    sentiment_store = config['queries']['sentiment_store']
    sentiment_analyses = config['queries']['sentiment_analysis']
    factor_store = config['queries']['factor_store']
    factors_analysis = config['queries']['factors_analysis']
    effect_store = config['queries']['effect_store']
    effect_analyses = config['queries']['effect_analysis']
    
    func_dict = dict(inspect.getmembers(queries, inspect.isfunction))
    
    # Initialize results structure
    res = initialize_results(config['subjects'], sentiment_analyses, directions)
    save_res(res, config_file, timestamp)

    # High level sentiments analysis
    logging.info('swapping model grammar: json')
    model = None; time.sleep(10)
    model = load_model(model_props, grammar='json.gbnf')
    
    process_sentiments(subjects, sentiment_store, sentiment_analyses, no_links, 
                      func_dict, config_file, model_props, model, res, timestamp)
    
    # Directional factors analysis
    logging.info('swapping model grammar: none')
    model = None; time.sleep(10)
    model = load_model(model_props, grammar='')
    
    process_factors(subjects, directions, factor_store, factors_analysis,
                   no_factors, no_links, func_dict, config_file, model_props,
                   model, res, timestamp)

def initialize_results(subjects: List[str], sentiment_analyses: List[str], 
                      directions: List[str]) -> Dict[str, Any]:
    """Initialize the results dictionary structure."""
    res = {}
    for subject in subjects:
        res[subject] = {'sentiments': {}, 'factors': {}}
        for sentiment_analysis in sentiment_analyses:
            res[subject]['sentiments'][sentiment_analysis] = {}
        for direction in directions:
            res[subject]['factors'][direction] = {}
    return res

def process_sentiments(subjects: List[str], sentiment_store: str, 
                      sentiment_analyses: List[str], no_links: int,
                      func_dict: Dict[str, Any], config_file: str,
                      model_props: Dict[str, Any], model: Any,
                      res: Dict[str, Any], timestamp: str) -> None:
    """Process sentiment analysis for all subjects."""
    for subject in subjects:
        analysis_id = subject.replace(' ', '_')
        vs = query_bing_and_create_vector_store(
            func_dict[sentiment_store], subject, '', '',
            no_links, config_file, analysis_id, 'sentiment_search'
        )
        
        for sentiment_analysis in sentiment_analyses:
            sentiment = llm_analyse(
                subject, '', func_dict[sentiment_analysis],
                '', no_links, vs, model_props, config_file, model
            )
            res[subject]['sentiments'][sentiment_analysis] = sentiment
            save_res(res, config_file, timestamp)

def process_factors(subjects: List[str], directions: List[str],
                   factor_store: str, factors_analysis: str,
                   no_factors: int, no_links: int, func_dict: Dict[str, Any],
                   config_file: str, model_props: Dict[str, Any],
                   model: Any, res: Dict[str, Any], timestamp: str) -> None:
    """Process factor analysis for all subjects."""
    for subject in subjects:
        analysis_id = subject.replace(' ', '_')
        for direction in directions:
            vs = query_bing_and_create_vector_store(
                func_dict[factor_store], subject, '',
                direction, no_links, config_file,
                analysis_id, direction + '_search'
            )
            
            factors = find_factors(
                subject, func_dict[factors_analysis],
                direction, no_factors, no_links,
                vs, model_props, config_file, model
            )
            
            for factor in factors:
                if is_not_a_number(factor):
                    res[subject]['factors'][direction][factor] = {}
                    save_res(res, config_file, timestamp)

def find_factors(subject: str, query: str, direction: str, no_factors: int, 
                no_links: int, vectorstore_faiss: Any, model_props: Dict[str, Any],
                config_file: str, model: Any) -> List[str]:
    """Find factors for a given subject and direction."""
    q, t = query()
    q = parameterise(q, subject, '', no_factors, direction)
    response = load_context_and_run_llm(q, t, model_props, '', vectorstore_faiss, config_file, model)
    res = parse_string_to_list(response)
    return res

def llm_analyse(subject: str, factor: str, query: str, direction: str, 
               no_links: int, vectorstore_faiss: Any, model_props: Dict[str, Any],
               config_file: str, model: Any) -> Dict[str, Any]:
    """Run LLM analysis for a given subject and factor."""
    q, t = query()
    q = parameterise(q, subject, factor, '', direction)
    response = load_context_and_run_llm(q, t, model_props, 'json.gbnf', vectorstore_faiss, config_file, model)
    res = parse_json_result(response, direction)
    return res

def query_bing_and_create_vector_store(query: str, subject: str, factor: str, 
                                      direction: str, no_links: int, config_file: str,
                                      analysis_id: str, prefix: str) -> Any:
    """Query Bing and create a vector store for a given subject and factor."""
    bs = query()
    bs = (parameterise(b, subject, factor, no_links, direction) for b in bs)
    search_bing_get_links_and_save(bs, no_links, config_file, analysis_id, prefix)
    vs = create_vector_db(config_file, analysis_id, prefix)
    return vs

def parameterise(inp: str, subject: str = '', factor: str = '', no_factors: str = '', 
                direction: str = '') -> str:
    """Parameterise a query string."""
    inp = inp.replace('{subject}', subject)
    inp = inp.replace('{factor}', factor)
    inp = inp.replace('{no_factors}', str(no_factors))
    inp = inp.replace('{direction}', direction)
    return inp

def is_not_a_number(factor: str) -> bool:
    """Check if a factor is not a number."""
    try:
        float(factor)
        return False  # not a valid factor if it can be converted to a float
    except ValueError:
        pass
    return True

def parse_string_to_list(answer: Dict[str, Any]) -> List[str]:
    """Parse a string to a list."""
    res = answer['result'].replace('"','')
    ls = res.split('\n')
    ls2 = []
    for l in ls:
        l = l.strip()
        if len(l) > 0:
            if l[0].isnumeric():
                l = l[2:].strip()
            elif l[0] == '-':
                l = l[2:].strip()
            elif l[0] == '*':
                l = l[2:].strip()
            ls2.append(l)
    for x in range(len(ls2)):
        ls2[x] = ls2[x].strip()
        ls2[x] = ls2[x].replace('.','')
        ls2[x] = ls2[x].replace(',','')
    return ls2

def save_res(res: Dict[str, Any], config_file: str, timestamp: str) -> None:
    """Save the results to a file."""
    path = os.path.join(os.getcwd(),'output', config_file, 'data')
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, 'signals.json')
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
        logging.info('saved result file to: ' + fname)
