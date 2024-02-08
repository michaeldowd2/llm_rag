import json
import os
import sys
import numpy as np
import time
import logging
import traceback

def parse_signals(config_file, config, timestamp):
    llm_output_folder = os.path.join(os.getcwd(),'output',config_file,'data')
    if not os.path.exists(llm_output_folder):
        print('no output data')
        return
    files = os.listdir(llm_output_folder)
    res = {'labels':[], 'datasets_ave_sig':[], 'datasets_sig_cnt':[], 'datasets_nz_sigs':[]} 
    securities = []
    for file in files:
        res['labels'].append(file.replace('json',''))
        file_path = os.path.join(llm_output_folder, file)
        with open(file_path) as json_file:
            try:
                data = json.load(json_file)
            except:
                print('error parsing: ' + file_path)
                json.load(json_file)
            for security in data:
                if security not in securities:
                    securities.append(security)

    for security in securities:   
        ave_sigs, sig_cnts, nz_sigs = [], [], []
        
        for file in files:
            file_path = os.path.join(llm_output_folder, file)
            with open(file_path) as json_file:
                try:
                    data = json.load(json_file)
                except:
                    print('error parsing: ' + json_file)
                    json.load(json_file)
                non_zero_signals, all_signals = 0, []
                if security in data and 'sentiments' in data[security]:
                    for sentiment in data[security]['sentiments']:
                        if type(data[security]['sentiments'][sentiment]) is dict and 'answer' in data[security]['sentiments'][sentiment]:
                            answer = str(data[security]['sentiments'][sentiment]['answer'])
                            numeric = parse_answer(answer)
                            if numeric != 0:
                                non_zero_signals += 1
                            all_signals.append(numeric)

            if len(all_signals)>0:
                ave_sigs.append(sum(all_signals) / len(all_signals))
            else:
                ave_sigs.append(0)
            sig_cnts.append(len(all_signals))
            nz_sigs.append(non_zero_signals)
            
        res['datasets_ave_sig'].append({'label':security, 'data':ave_sigs})
        res['datasets_sig_cnt'].append({'label':security, 'data':sig_cnts})
        res['datasets_nz_sigs'].append({'label':security, 'data':nz_sigs})

    SaveRes(res, config_file, timestamp)

def SaveRes(res, config_file, timestamp):
    path = os.path.join(os.getcwd(), 'output', config_file)
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info('created res output path at: ' + path)
    fname = os.path.join(path, 'parsed_signals.json')
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
        logging.info('saved result file to: ' + fname)

def parse_answer(answer):   
    answer = answer.lower()
    answer = answer.replace('its','')
    answer = answer.replace('to', '')
    answer = answer.replace('continue', '')
    answer = answer.replace('remain','')
    answer = answer.replace('stay','')
    answer = answer.replace('reach','')
    answer = answer.replace('significantly','')
    answer = answer.replace('experience','')
    answer = answer.replace('a','')
    answer = answer.replace('significant','')
    answer = answer.replace('some','')
    answer = answer.replace('cautiously','')
    answer = answer.replace('slightly','')
    answer = answer.replace('see','')
    answer = answer.replace('major','')
    answer = answer.replace('against the us dollar','')
    answer = answer.strip()
    numeric = 0
    #these ones get inverted based on direction
    if answer in ['yes']:
        numeric = 1
    elif answer in ['extremely likely']:
        numeric = 0.9
    elif answer in ['very likely']:
        numeric = 0.75
    elif answer in ['likely']:
        numeric = 0.6
    elif answer in ['possible']:
        numeric = 0.45
    elif answer in ['no','unlikely']:
        answer = 0
      
    if answer in ['increase','rally','appreciate','upward momentum','growth','bullish trend','ascension','optimistic','gains','upward trajectory','trending upward','recover','gain strength','rebound','grow','rise','rising','upward','upward trend','recent climb','strengthen','strong','recovering','positive','bullish','new highs','new high','record highs','record high','new heights','record heights']:
        numeric = 1
    elif answer in ['decrease','depreciate','shed value','drop','decline','crash','trade at lower levels','declining','drop','downward trajectory','lose strength','decreasing','weakening','decline further','fall','falling','downward','downward trend','tank','weaken','weak','negative','bearish','new lows','new low','record lows','record low']:
        numeric = -1
    elif answer in ['fluctuate','volatile','stable']:
        numeric = 0
    return numeric

