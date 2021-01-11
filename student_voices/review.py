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

    
    
def get_topic_summary(results_dir, tn, rng, config, setting):

    name = rng+'_'+str(tn)+'_'+setting+'_'+config

    # Get the relevant coherence scores 
    coherence = bn.decompress_pickle(results_dir+'/complete_coherence.pbz2')
    topic_coherence = [c for c in coherence[rng][config][setting] if c[0]==tn][0][1][0]
    # Get the relevant reviews 
    reviews = bn.decompress_pickle(results_dir+'/LDAdistributions/Vec_'+name+'.pbz2')
    review_dist = list(reviews['Dominant_Topic'].value_counts().sort_index().values)
    # Get the relevant descriptive words 
    description = pd.read_csv(results_dir+'/LDAdescriptions/Des_'+name+'.csv', header=None)
    description = description[['words' in r for r in description[0]]]
    descriptions = []
    for idx, row in description.iterrows(): descriptions.append(' '.join(row.values))    


    model_info = pd.DataFrame({'Coherence':topic_coherence, 'NumReviews':review_dist, 'Descriptions':descriptions}).reset_index()
    model_info['index']=model_info['index']+1
    model_info = model_info.set_index('index')
    
    return model_info
    