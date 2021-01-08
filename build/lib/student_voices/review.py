# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:04:45 2021

@author: Computer
"""

import glob 
import re 
import os
import pandas as pd 

def get_models_completed(path):
    model_directories = [x[0] for x in os.walk(path)][1:]
    models_done = {} 
    models_done['setting'], models_done['config'], models_done['range'], models_done['n_topics'] = [], [], [], [] 
    
    for directory in model_directories: 
        model_names = glob.glob(directory+'/*.lda')
        for mn in model_names: 
            setting, config = directory.split('/')[-1].split('_')
            rng = re.findall('\[.*\)', mn)[0]
            n_topics = int(re.findall('\)([0-9]*)\.', mn)[0])
            
            models_done['setting'].append(setting)
            models_done['config'].append(config)
            models_done['range'].append(rng)
            models_done['n_topics'].append(n_topics)
    
    models_done = pd.DataFrame(models_done)
    
    return models_done 
    
    
    