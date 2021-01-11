import os, sys
sys.path.append('../')

from student_voices import sv_utils as bn
from student_voices import lda_analysis as ld
from student_voices import modeling_tools as mt 
from student_voices import review

import numpy as np
import argparse, time
import pandas as pd

from path import Path 
root = Path(os.path.dirname(os.path.abspath(__file__)))


def run_coherence_analysis_ff(data_dir, model_dir, results_dir):
    
    # import the data if need be
    data = bn.decompress_pickle(root+'/data/review_stats.pbz2')

    models_completed = review.get_models_completed(model_dir)

    # We don't want to be loading the data several times so we will go through the list of models in order, one config at a time
    configs = list(models_completed['Config'].unique())        

    coherence_guide = {}
    ranked_results = pd.DataFrame()
    
    results_directory = results_dir+'/'

    st = time.time()

    for config in configs: 
        text, stem_map, lemma_map, phrase_frequencies = bn.decompress_pickle(data_dir+'/cleaned_docs_'+config+'.pbz2')

        config_models = models_completed[models_completed['Config']==config]
        # We'll go through the ranges in order as well because we'll have to laod the corresponding by_rating_range             
        current_ranges = list(config_models['Range'].unique())

        for rng in current_ranges: 
            # Select the indices grouping to load based on the range 
            if rng == '[0, 60)': corpus_group = 'B'
            elif rng == '[0, 65)': corpus_group = 'B'
            else: corpus_group = 'A'

            if corpus_group == 'A': 
                range_indices = bn.loosen(root + '/data/by_rating_range.pickle')
            elif corpus_group == 'B': 
                range_indices = bn.loosen(root + '/data/by_rating_range_2.pickle')
            elif corpus_group == 'C':
                range_indices = bn.loosen(root + '/data/by_rating_range_3.pickle')
    
            # create a list of each range 
            ranges = list(np.sort(list(range_indices.keys())))

            range_models = config_models[config_models['Range']==rng]
            
            # Now we want to go through the settings in order as well and compare the models for all the number of topics in each setting
            settings = list(range_models['Setting'].unique())

            for setting in settings: 
                setting_models = range_models[range_models['Setting']==setting]
                n_topics_list = sorted(list(setting_models['N_Topics'].unique()))
        
                model_directory = model_dir+'/'+setting+'_'+config+'/'

                # import hardcoded lda paramter dictionary 
                lda_parameters =ld.hardcoded_lda_parameters(ranges, range_indices, numtopics=None, custom_numtopics=n_topics_list)

                indices = data.loc[range_indices[rng], 'Review_Length']
                    
                # filter the training corpus by review length and save the length 
                docs = [text[idx] for idx in indices[indices>lda_parameters[setting][rng]['review_length']].index]
                
                # Load the trained models that we're going to compare 
                trained_models = ld.load_models(model_directory,setting+'_'+config+'_'+rng, n_topics_list) # this list is the topic numbers that were tried. They are hardcoded but could probably be made an argument later
            
                corpus, dictionary, literal_dictionary, id2word, word2id = mt.set_dictionary(docs,
                                                                                         lda_parameters[setting][rng]['nbelow'],
                                                                                         lda_parameters[setting][rng]['nabove'])
                # get the ranked coherence models and the coherence
                ranked, coherences = ld.determine_coherence(trained_models, dictionary, docs)
        
                # store the ranked results in a dataset
                ranked_scores = pd.DataFrame(ranked, columns=['num_topics','ave_coherence_score'])
                ranked_scores['range']=str(rng)
                ranked_scores['setting']=str(setting)
                ranked_scores['config']=str(config)
                ranked_results = ranked_results.append(ranked_scores)    
                
                # store the full coherence scores per topic in a separate dictionary object
                if rng not in coherence_guide: coherence_guide[rng]={} 
                if config not in coherence_guide[rng]: coherence_guide[rng][config]={}
                if setting not in coherence_guide[rng][config]: coherence_guide[rng][config][setting]={}
            
                coherence_guide[rng][config][setting] = [(k,coherences[k]) for k in coherences]
        
                print('Estimating coherence for range '+str(rng)+', setting '+str(setting)+', config '+str(config)+' took '+str(time.time()-st))
                
    bn.compressed_pickle(results_directory+'/complete_coherence', coherence_guide)
        
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('FINISHED')


# Needs to be updated to exclude the numtopics and corpus group args 
#def run_coherence_analysis(setting, config, numtopics, data_dir, model_dir, results_dir, corpus_group):
#
#    text, stem_map, lemma_map, phrase_frequencies = bn.decompress_pickle(data_dir+'/cleaned_docs_'+config+'.pbz2')
#
#    model_directory = model_dir+'/'+setting+'_'+config+'/'
#
#    results_directory = results_dir+'/'+setting+'_'+config+'/'
#
#    if not os.path.exists(results_directory): 
#        print('Creating directory ', str(results_directory))
#        os.mkdir(results_directory)
#
#    if corpus_group == 'A': 
#        range_indices = bn.loosen(root + '/data/by_rating_range.pickle')
#    elif corpus_group == 'B': 
#        range_indices = bn.loosen(root + '/data/by_rating_range_2.pickle')
#
#    # import the data if need be
#    data = bn.decompress_pickle(root+'/data/review_stats.pbz2')
#    # create a list of each range 
#    ranges = list(np.sort(list(range_indices.keys())))
#    # import hardcoded lda paramter dictionary 
#    lda_parameters =ld.hardcoded_lda_parameters(ranges, range_indices, numtopics)
#
#    coherence_guide = {}
#    ranked_results = pd.DataFrame()
#
#    st = time.time()
#
#    for rng in ranges: 
#
#        indices = data.loc[range_indices[rng], 'Review_Length']
#            
#        # filter the training corpus by review length and save the length 
#        docs = [text[idx] for idx in indices[indices>lda_parameters[setting][rng]['review_length']].index]
#        
#        # Load the trained models that we're going to compare 
#        trained_models = ld.load_models(model_directory,setting+'_'+config+'_'+rng, list(range(3,30,3))) # this list is the topic numbers that were tried. They are hardcoded but could probably be made an argument later
#    
#        corpus, dictionary, literal_dictionary, id2word, word2id = mt.set_dictionary(docs,
#                                                                                 lda_parameters[setting][rng]['nbelow'],
#                                                                                 lda_parameters[setting][rng]['nabove'])
#        # get the ranked coherence models and the coherence
#        ranked, coherences = ld.determine_coherence(trained_models, dictionary, docs)
#
#        # store the ranked results in a dataset
#        ranked_scores = pd.DataFrame(ranked, columns=['num_topics','ave_coherence_score'])
#        ranked_scores['range']=str(rng)
#        ranked_scores['setting']=str(setting)
#        ranked_scores['config']=str(config)
#        ranked_results = ranked_results.append(ranked_scores)    
#        
#        # store the full coherence scores per topic in a separate dictionary object
#        if rng not in coherence_guide: coherence_guide[rng]={} 
#        if config not in coherence_guide[rng]: coherence_guide[rng][config]={}
#        if setting not in coherence_guide[rng][config]: coherence_guide[rng][config][setting]={}
#    
#        coherence_guide[rng][config][setting] = [(k,coherences[k]) for k in coherences]
#
#        print('Estimating coherence for range '+str(rng)+', setting '+str(setting)+', config '+str(config)+' took '+str(time.time()-st))
#
#    ranked_results.to_csv(results_directory+'/ranked_coherence_'+str(setting)+'_'+str(config)+'.csv')        
#    bn.compressed_pickle(results_directory+'/full_coherence_'+str(setting)+'_'+str(config), coherence_guide)




'''
The ranked result is a list that allows you to print stuff like this: 
    
    num_topics=25:	0.5049
    num_topics=10:	0.4769
    num_topics=40:	0.4630
    num_topics=55:	0.4539
    num_topics=85:	0.4361
    num_topics=70:	0.4321
    num_topics=100:	0.4308
    
    Best: 25

Where each item of the list is a tuple with (num_topics, coherence_score)
It is ranked such that ranked[0][0] is the most coherent number of topics and ranked[0][1] is its score. 

Coherences returns the full coherence results as a dictionary which includes 
the coherence scores estimated for every topic in every range-setting-config run. 
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Launch spot instance')
    parser.add_argument('-cp', '--configpath', help='Path to configuration data', required=True)
    parser.add_argument('-md', '--modeldir', help='Path to save the models', required=True)
    parser.add_argument('-rd', '--resultsdir', help='Path to save the results', required=True)
    #parser.add_argument('-c', '--configuration', help='Configuration (A1,B1,C1,...)', default=None)
    #parser.add_argument('-s', '--setting', help='LDA parameter setting to use from hardcoded options', default=None)
    #parser.add_argument('-nt', '--numtopics', help='Option for the number of topics: A,B,C,...', default=None)
    #parser.add_argument('-cg', '--corpusgrouping', help='Option for the corpus grouping to pick: 1, 2, 3,...', default=None)

    args = parser.parse_args()

    config_path = args.configpath
    model_dir = args.modeldir
    results_dir = args.resultsdir
    
    run_coherence_analysis_ff(config_path, model_dir, results_dir)
    
#    # This approach will be deprecated 
#    setting = args.setting
#    config = args.configuration
#    numtopics = args.numtopics
#    corpus_group = args.corpusgrouping
#    assert setting is not None 
#    assert config is not None 
#    assert numtopics is not None 
#    assert corpus_group is not None
#    run_coherence_analysis(setting, config, numtopics, config_path, model_dir, results_dir, corpus_group)

