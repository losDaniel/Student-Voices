import os, sys
sys.path.append('../')

from student_voices import sv_utils as bn
from student_voices import lda_analysis as ld
from student_voices import modeling_tools as mt 
from student_voices import visuals as vs 

import numpy as np
import argparse, time 

from path import Path 
root = Path(os.path.dirname(os.path.abspath(__file__)))


def setup_context(setting, config, ntopics_option, full_path, model_dir, clean_dir, corpus_group):
    
    full_text = bn.decompress_pickle(full_path+'/full_review_text.pbz2')
    
    text, stem_map, lemma_map, phrase_frequencies = bn.decompress_pickle(clean_dir+'/cleaned_docs_'+config+'.pbz2')

    if corpus_group == 'A': 
        range_indices = bn.loosen(root + '/data/by_rating_range.pickle')
    elif corpus_group == 'B': 
        range_indices = bn.loosen(root + '/data/by_rating_range_2.pickle')
    else: 
        raise "Please submit valid corpus group"

    # create a list of each range 
    ranges = list(np.sort(list(range_indices.keys())))

    # import the data if need be
    data = bn.decompress_pickle(root+'/data/review_stats.pbz2')

    # import hardcoded lda paramter dictionary 
    lda_parameters =ld.hardcoded_lda_parameters(ranges, range_indices, ntopics_option)

    return lda_parameters, data, ranges, range_indices, text, full_text


def setup_rng_context(rng, range_indices, data, setting, config, lda_parameters, text, full_text, model_dir):

    # pull the column of review length
    indices = data.loc[range_indices[rng],'Review_Length']
    # keep the indices of the rows with review length > than the minimum length
    filtered_index = indices[indices>lda_parameters[setting][rng]['review_length']].index
    
    # filter the training corpus by review length and save the length 
    docs = [text[idx] for idx in filtered_index]
    # get original reviews for the same documents 
    fulldocs = [full_text[idx] for idx in filtered_index]
    
    model_directory = model_dir+'/'+setting+'_'+config+'/'

    # load the models 
    trained_models = ld.load_models(model_directory,setting+'_'+config+'_'+rng,lda_parameters[setting][rng]['ntrange'])

    # get the dictionary 
    corpus, dictionary, literal_dictionary, id2word, word2id = mt.set_dictionary(docs,
                                                                             lda_parameters[setting][rng]['nbelow'],
                                                                             lda_parameters[setting][rng]['nabove'])

    return trained_models, dictionary, docs, fulldocs, corpus


def run_lda_visualization(setting, 
                          config, 
                          ntopics_option, 
                          corpus_group,
                          topic_num, 
                          num_words,
                          rng, 
                          clean_dir, 
                          graph_path, 
                          full_path, 
                          model_dir, 
                          des_dir,
                          vec_dir, 
                          ):
    
    lda_parameters, data, ranges, range_indices, text, full_text = setup_context(setting, config, ntopics_option, full_path, model_dir, clean_dir, corpus_group)
        
    trained_models, dictionary, docs, fulldocs, corpus = setup_rng_context(rng, range_indices, data, setting, config, lda_parameters, text, full_text, model_dir)

    # get the most coherent model 
    model = trained_models[topic_num]

    # specify paths to save the results  
    lda_viz_path = graph_path+'/Viz_'+rng+'_'+str(topic_num)+'_'+setting+'_'+config+'.html'
    topic_des_path = des_dir+'/Des_'+rng+'_'+str(topic_num)+'_'+setting+'_'+config+'.csv'
    topic_vec_path = vec_dir+'/Vec_'+rng+'_'+str(topic_num)+'_'+setting+'_'+config # no extension because we will compress
    
    # Create and save the topic pyLDAvis HTML topic visualization 
    vs.save_topic_visualization(model, docs, dictionary, lda_viz_path)
    
    # Save the top words for each topic and their coefficients 
    ld.write_lda_descriptions(topic_des_path, model, num_words)

    # Get main topic in each document
    sentence_topics_df = ld.get_sentence_topics(model, corpus, fulldocs, path=topic_vec_path)
    
    return sentence_topics_df



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Launch spot instance')
    parser.add_argument('-c', '--configuration', help='Configuration (A1,B1,C1,...)', required=True)
    parser.add_argument('-s', '--setting', help='LDA parameter setting to use from hardcoded options', required=True)
    parser.add_argument('-cp', '--cleanpath', help='Path to configuration data', required=True)
    parser.add_argument('-md', '--modeldir', help='Path to save the models', required=True)
    parser.add_argument('-nt', '--numtopics', help='Option for the number of topics: A,B,C,...', required=True)
    parser.add_argument('-cg', '--corpusgrouping', help='Option for the corpus grouping to pick: 1, 2, 3,...', required=True)
    parser.add_argument('-fp', '--fullpath', help='path to full_review_text.pbz2', required=True)
    parser.add_argument('-nw', '--numwords', help='number of words to include in descriptions', required=True)
    parser.add_argument('-gp', '--graphpath', help='path to save the graphs at', required=True)
    parser.add_argument('-dd', '--desdir', help='Path to save the topic descriptions', required=True)
    parser.add_argument('-vd', '--vecdir', help='Path to save the topic vectors', required=True)

    args = parser.parse_args()

    config = args.configuration
    clean_dir = args.cleanpath
    model_dir = args.modeldir
    setting = args.setting
    ntopics_option = args.numtopics
    corpus_group = args.corpusgrouping
    full_path = args.fullpath
    num_words = args.numwords
    graph_path = args.graphpath
    des_dir = args.desdir 
    vec_dir = args.vecdir
        
    # topic_num - run this for each topic number in the hardcodes 
    from student_voices.lda_analysis import get_num_topic_option
    num_topics_options = get_num_topic_option()
    number_of_topics_to_try = num_topics_options[ntopics_option]
    
    if corpus_group == 'A': 
        range_indices = bn.loosen(root + '/data/by_rating_range.pickle')
    elif corpus_group == 'B': 
        range_indices = bn.loosen(root + '/data/by_rating_range_2.pickle')
    else: 
        raise "Please submit valid corpus group"

    # create a list of each range 
    ranges = list(np.sort(list(range_indices.keys())))

    for topic_num in number_of_topics_to_try:
        for rng in ranges: 
        
            st = time.time() 
            print('Beginning visualization for ntopics: '+str(topic_num)+', range: '+str(rng))
            run_lda_visualization(setting, 
                          config, 
                          ntopics_option, 
                          corpus_group,
                          topic_num, 
                          num_words,
                          rng, 
                          clean_dir, 
                          graph_path, 
                          full_path, 
                          model_dir, 
                          des_dir,
                          vec_dir)

            print('Completed. Duration: '+str(time.time()-st))