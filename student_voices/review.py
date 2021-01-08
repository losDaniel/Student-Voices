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
    models_done['Setting'], models_done['Config'], models_done['Range'], models_done['N_Topics'] = [], [], [], [] 
    
    for directory in model_directories: 
        model_names = glob.glob(directory+'/*.lda')
        for mn in model_names: 
            setting, config = directory.split('/')[-1].split('_')
            setting = setting.split('\\')[-1]
            rng = re.findall('\[.*\)', mn)[0]
            n_topics = int(re.findall('\)([0-9]*)\.', mn)[0])
            
            models_done['Setting'].append(setting)
            models_done['Config'].append(config)
            models_done['Range'].append(rng)
            models_done['N_Topics'].append(n_topics)
    
    models_done = pd.DataFrame(models_done)
    
    return models_done 
    
    
    