"""
Web utilities for fetching and processing URLs.
Contains functions for making web requests and handling Bing search results.
"""
import requests
import logging
import time
import os
import traceback
from bs4 import BeautifulSoup
import lxml
import json
from typing import List

def request_url_and_save(url: str, save: bool, config_file: str, analysis_id: str, doc_prefix: str, doc_id: int) -> None:
    """
    Fetch URL content and optionally save to file.
    
    Args:
        url: Target URL to fetch
        save: Whether to save the content
        config_file: Configuration file name
        analysis_id: ID for the analysis
        doc_prefix: Prefix for the document
        doc_id: Document ID number
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.19582" 
        }
        
        logging.info('sending request for: ' + url)
        response = requests.get(url, headers=headers, timeout=60).text
        time.sleep(2)
        
        if save:
            path = os.path.join(os.getcwd(), 'output', config_file, 'html', analysis_id)
            path = os.path.join(path, doc_prefix)
            if not os.path.exists(path):
                os.makedirs(path)
                logging.info('created html output path at: ' + path)
            fname = os.path.join(path, str(doc_id) + '.html')
            with open(fname, "w", encoding="utf-8") as f:
                f.write(response)
                logging.info('saved html file to: ' + fname)
    except Exception as e:
        logging.exception("error requesting url")
        logging.exception(traceback.format_exc())
        raise Exception(e)

def search_bing_get_links_and_save(queries: List[str], no_links: int, config_file: str, analysis_id: str, doc_prefix: str) -> None:
    """
    Search Bing for queries and save results.
    
    Args:
        queries: List of search queries
        no_links: Number of links to process per query
        config_file: Configuration file name
        analysis_id: ID for the analysis
        doc_prefix: Prefix for saving documents
    """
    doc_id = 0
    for query in queries:
        try:
            # Note: This is a placeholder for Bing search implementation
            # In practice, you would implement actual Bing search API calls here
            logging.info('searching bing for: ' + query)
            # Simulate getting URLs from Bing
            urls = []  # Would come from Bing API
            
            for url in urls[:no_links]:
                actual_url = bing_to_actual_url(url)
                request_url_and_save(actual_url, True, config_file, analysis_id, doc_prefix, doc_id)
                doc_id += 1
                
        except Exception as e:
            logging.exception("error searching bing")
            logging.exception(traceback.format_exc())
            continue

def bing_to_actual_url(url: str) -> str:
    """
    Convert Bing redirect URL to actual URL.
    
    Args:
        url: Bing redirect URL
        
    Returns:
        str: Actual destination URL
    """
    try:
        response = requests.get(url, allow_redirects=False)
        if response.status_code in [301, 302]:
            return response.headers['Location']
    except:
        pass
    return url
