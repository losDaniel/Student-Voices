import os, sys
sys.path.append('../')

from student_voices import sv_utils as bn
from student_voices import lda_analysis as ld
from student_voices import modeling_tools as mt 

import numpy as np
import argparse, time
import pandas as pd

from path import Path 
root = Path(os.path.dirname(os.path.abspath(__file__)))




    # specify the number of words  you want to display when describing each topic 
    num_words = 15


def run_coherence_analysis(setting, config, data_dir, model_dir, results_dir):
    

    # same as above, we want to loop through the experiment and create the graphs as needed 
    full_text = bn.decompress_pickle(os.getcwd()+'/data/full_review_text.pbz2')   #< modified
    
    text, stem_map, lemma_map, phrase_frequencies = bn.decompress_pickle(data_dir+'/cleaned_docs_'+config+'.pbz2')

    model_directory = model_dir+'/'+setting+'_'+config+'/'

    results_directory = results_dir+'/'+setting+'_'+config+'/'

    if not os.path.exists(results_directory): 
        print('Creating directory ', str(results_directory))
        os.mkdir(results_directory)

    # import the data if need be
    data = bn.decompress_pickle(root+'/data/review_stats.pbz2')
    # import the range indices 
    range_indices = bn.loosen(root + '/data/by_rating_range.pickle')
    # create a list of each range 
    ranges = list(np.sort(list(range_indices.keys())))
    # import hardcoded lda paramter dictionary 
    lda_parameters =ld.hardcoded_lda_parameters(ranges, range_indices)

    coherence_guide = {}
    ranked_results = pd.DataFrame()

    st = time.time()

    for rng in ranges: 

        indices = data.loc[range_indices[rng], 'Review_Length']

        # keep the indices of the rows with review length > than the minimum length
        filtered_index = indices[indices>lda_parameters[setting][rng]['review_length']].index        
        # filter the training corpus by review length and save the length 
        docs = [text[idx] for idx in filtered_index]
        # get original reviews for the same documents 
        fulldocs = [full_text[idx] for idx in filtered_index]
       
        # Load the trained models that we're going to compare 
        trained_models = ld.load_models(model_directory,setting+'_'+config+'_'+rng, list(range(3,30,3))) # this list is the topic numbers that were tried. They are hardcoded but could probably be made an argument later
    
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

    ranked_results.to_csv(results_directory+'/ranked_coherence_'+str(setting)+'_'+str(config)+'.csv')        
    bn.compressed_pickle(results_directory+'/full_coherence_'+str(setting)+'_'+str(config), coherence_guide)






    

    # extract the coherence score data
    coherence_scores=list(zip(re.findall('[0-9]+\.*[0-9]*', row['coherence_scores'])[::2],
                              re.findall('[0-9]+\.*[0-9]*', row['coherence_scores'])[1::2]))

    # get the topic num with the top coherence score 
    best_topic_num = int(coherence_scores[0][0])

    # get the most coherent model 
    model = trained_models[best_topic_num]

    # specify paths to save the results  
    lda_viz_path = os.getcwd()+'/graphs/LDA Graphs/Viz_'+rng+'_'+str(best_topic_num)+'_'+setting+'_'+config+'.html'
    topic_des_path = os.getcwd()+'/results/LDA Descriptions/Des_'+rng+'_'+str(best_topic_num)+'_'+setting+'_'+config+'.csv'
    topic_vec_path = os.getcwd()+'/results/LDA Distributions/Vec_'+rng+'_'+str(best_topic_num)+'_'+setting+'_'+config # no extension because we will compress
    
    # Create and save the topic pyLDAvis HTML topic visualization 
    vs.save_topic_visualization(model, docs, dictionary, lda_viz_path)
    
    # Save the top words for each topic and their coefficients 
    ld.write_lda_descriptions(topic_des_path, model, num_words)

    # Get main topic in each document
    sentence_topics_df = ld.get_sentence_topics(model, corpus, fulldocs, path=topic_vec_path)


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
    parser.add_argument('-c', '--configuration', help='Configuration (A1,B1,C1,...)', required=True)
    parser.add_argument('-s', '--setting', help='LDA parameter setting to use from hardcoded options', required=True)
    parser.add_argument('-cp', '--configpath', help='Path to configuration data', required=True)
    parser.add_argument('-md', '--modeldir', help='Path to save the models', required=True)
    parser.add_argument('-rd', '--resultsdir', help='Path to save the results', required=True)

    args = parser.parse_args()

    config = args.configuration
    config_path = args.configpath
    model_dir = args.modeldir
    setting = args.setting
    results_dir = args.resultsdir
    
    run_coherence_analysis(setting, config, config_path, model_dir, results_dir)


