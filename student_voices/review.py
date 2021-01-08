import sys
sys.path.append('../')

from student_voices import sv_utils as bn

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


def get_scored_unscored_models(results_path, models_path):
    full_coherence_data = bn.loosen(results_path+'/full_coherence_data.pickle')
    completed_models = get_models_completed(models_path)
    models_scoring = completed_models.merge(full_coherence_data[['Setting','Config','Range','N_Topics']], on=['Setting','Config','Range','N_Topics'], how='outer', indicator=True)

    print(models_scoring['_merge'].value_counts())
    models_scored = models_scoring[models_scoring['_merge']=='both']
    models_unscored = models_scoring[models_scoring['_merge']=='left_only']    
    
    return models_scored, models_unscored 

    
    
    