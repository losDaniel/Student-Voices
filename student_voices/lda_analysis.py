import os, sys
sys.path.append('../')

from student_voices import sv_utils as bn 
from student_voices import modeling_tools as mt 

from collections import OrderedDict
from gensim import models

import time, multiprocessing
import pandas as pd

cpu_count = multiprocessing.cpu_count() - 1
print("Available Cores: %d" % cpu_count)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-# MODELING  FUNCTIONS #-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


def get_num_topic_option():
    
    num_topics_options = {} 
    num_topics_options['A'] = list(range(3,30,3))
    num_topics_options['B'] = [16, 17, 19, 20]
    num_topics_options['C'] = [22, 23, 25, 26]
    num_topics_options['D'] = [4, 5, 7, 8, 10, 11]
    num_topics_options['E'] = list(range(5,12))+list(range(12,33,4))
    num_topics_options['F'] = [12, 15, 16, 17, 18] 
    num_topics_options['G'] = [19, 20, 21, 22, 23] 
    num_topics_options['H'] = [24, 25, 26, 27]
    num_topics_options['I'] = [25]

    return num_topics_options    

def hardcoded_lda_parameters(ranges, range_indices, numtopics=None, custom_numtopics=None): 

    lda_parameters={}

    if custom_numtopics is None:     
        assert numtopics is not None 
        num_topics_options = get_num_topic_option()
        number_of_topics_to_try = num_topics_options[numtopics]
    else: 
        number_of_topics_to_try = custom_numtopics
    
    lda_parameters['LDA1']={} 
    for rng in ranges: 
        corp_len = len(range_indices[rng])
    
        lda_parameters['LDA1'][rng]={}
        lda_parameters['LDA1'][rng]['ntrange'] = number_of_topics_to_try
        lda_parameters['LDA1'][rng]['review_length'] = 100
        lda_parameters['LDA1'][rng]['passes'] = 64
        lda_parameters['LDA1'][rng]['nbelow'] = 30
        lda_parameters['LDA1'][rng]['nabove'] = .4
        lda_parameters['LDA1'][rng]['corpus_length']=corp_len   
    
    if '[95, 101)' in lda_parameters['LDA1'].keys():
        # Increase the review length requirement for the highest category because of sheer volume. Its not necessary to keep a corpus that's so large.     
        lda_parameters['LDA1']['[95, 101)']['review_length'] = 175 

    
    lda_parameters['LDA2']={} 
    for rng in ranges: 
        corp_len = len(range_indices[rng])
    
        lda_parameters['LDA2'][rng]={}
        lda_parameters['LDA2'][rng]['ntrange'] = number_of_topics_to_try
        lda_parameters['LDA2'][rng]['review_length'] = 125
        lda_parameters['LDA2'][rng]['passes'] = 64
        lda_parameters['LDA2'][rng]['nbelow'] = 20
        lda_parameters['LDA2'][rng]['nabove'] = .4
        lda_parameters['LDA2'][rng]['corpus_length']=corp_len   
    
    if '[95, 101)' in lda_parameters['LDA2'].keys():
        # Increase the review length requirement for the highest category because of sheer volume. Its not necessary to keep a corpus that's so large.     
        lda_parameters['LDA2']['[95, 101)']['review_length'] = 200 


    lda_parameters['LDA3']={} 
    for rng in ranges: 
        corp_len = len(range_indices[rng])
    
        lda_parameters['LDA3'][rng]={}
        lda_parameters['LDA3'][rng]['ntrange'] = number_of_topics_to_try
        lda_parameters['LDA3'][rng]['review_length'] = 150
        lda_parameters['LDA3'][rng]['passes'] = 64
        lda_parameters['LDA3'][rng]['nbelow'] = 20
        lda_parameters['LDA3'][rng]['nabove'] = .4
        lda_parameters['LDA3'][rng]['corpus_length']=corp_len   
    
    if '[95, 101)' in lda_parameters['LDA3'].keys():
        # Increase the review length requirement for the highest category because of sheer volume. Its not necessary to keep a corpus that's so large.     
        lda_parameters['LDA3']['[95, 101)']['review_length'] = 225 

    
    lda_parameters['LDA4']={} 
    for rng in ranges: 
        corp_len = len(range_indices[rng])
    
        lda_parameters['LDA4'][rng]={}
        lda_parameters['LDA4'][rng]['ntrange'] = number_of_topics_to_try
        lda_parameters['LDA4'][rng]['review_length'] = 175
        lda_parameters['LDA4'][rng]['passes'] = 64
        lda_parameters['LDA4'][rng]['nbelow'] = 20
        lda_parameters['LDA4'][rng]['nabove'] = .4
        lda_parameters['LDA4'][rng]['corpus_length']=corp_len   
        
    if '[95, 101)' in lda_parameters['LDA4'].keys():
        # Increase the review length requirement for the highest category because of sheer volume. Its not necessary to keep a corpus that's so large.     
        lda_parameters['LDA4']['[95, 101)']['review_length'] = 250 

    
    return lda_parameters



# Train LDA topic models 
def train_ldas(docs, passes, ntrange, id2word, cores = cpu_count):
    '''
    MultiTrain LDA clusters with a range of ntrange clusters and return a dictionary of models
    docs - list of documents in bag of words format 
    dictionary - gensim dictionary item 
    passes - number of passes through the corpus for each model 
    ntrange - array number of topics to test. 
    cores - number of cores to be used 
    '''

    # create an ordered dictionary to store the results from each model
    trained_models = OrderedDict()
    
    print('Iterating over topic numbers')
    st = time.time()

    # we will train models for different numbers of topics and evaluate the coherence for each 
    for num_topics in ntrange: 
        
        st2 = time.time()
        
        print("Training LDA(k=%d)" % num_topics)
        # train the model on multiple cores
        lda = models.LdaMulticore(
            corpus = docs, id2word = id2word, num_topics = num_topics, 
            workers = cores, passes=passes, random_state = 2, eval_every = 1,
            alpha = 'asymmetric',
            decay=0.5, offset=64  # best params from Hoffman paper
        ) 
        
        # add it to the dictionary of trained models 
        trained_models[num_topics] = lda
        
        print('Finished in '+str(time.time()-st2)+' at '+str(time.time()-st))
    
    return trained_models



# run lda models for a corpus given a set of parameters 
def run_lda(docs, params):
    # keep track of time 
    st = time.time()
    # setup the text data with the specific model cleaning parameters 
    corpus, dictionary, literal_dictionary, id2word, word2id = mt.set_dictionary(docs, 
                                                                              nb=params['nbelow'], 
                                                                              na=params['nabove'])
    # train the lda models 
    trained_models = train_ldas(corpus, 
                                   params['passes'],
                                   params['ntrange'],
                                   id2word,
                                   cores=cpu_count)
    # print how long it took to train
    duration = time.time() - st
    print('Training all the models on the corpus took %s' % str(duration))
    
    return trained_models, corpus, dictionary, duration



def save_models(named_models, models_dir, name):
    '''Model specifically to save models in dictionary of models'''
    if not os.path.isdir(models_dir): 
        os.mkdir(models_dir)
    for num_topics, model in named_models.items():
        model_path = os.path.join(models_dir, name + '%d.lda' % num_topics)
        model.save(model_path, separately=False)


    
def load_models(models_dir, name, ntrange):
    trained_models = OrderedDict()
    for num_topics in ntrange:
        model_path = os.path.join(models_dir, name + '%d.lda' % num_topics)
        #print("Loading LDA(k=%d) from %s" % (num_topics, model_path))
        trained_models[num_topics] = models.LdaMulticore.load(model_path)

    return trained_models
    

    
'''    
# Guided LDA functions need to be worked on 
# -----------------------------------------

# turn a bow corpus in the document term matrix 
def bow2dtm(docs, dictionary): 
    
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    # every document will be a dictionary length vector 
    dtm = []
    for doc in corpus: 
        vec = np.zeros(len(dictionary))
        # get each word and count instance 
        for word in doc: 
            # the 'word'th position will be the frequency count 
            vec[word[0]] = int(word[1])
        dtm.append(vec)
    #dtm = np.array(dtm)

    return dtm 

    

def run_guidedlda(dconf, mconfig, ranges, seed_topic_list):
    # No multiprocessing for guided LDA
    model_directory = {} 

    total_time = 0 
    # for each rating range (corpus) train a series of LDA models and test out coherence. 
    for rng in ranges: 
        st = time.time()
        # instantiate the dictionaries 
        docs, stem_map, lemma_map, phrase_freq, dictionary, literal_dictionary, id2word, word2id = mt.setup_text_training_data(rng, 
                                                                                                                               dconf, 
                                                                                                                               mconfig['nbelow'], 
                                                                                                                               mconfig['nabove'])
        # assign topic numbers to seed words 
        seed_topics = {}
        for t_id, st in enumerate(seed_topic_list):
            for word in st:
                try:
                    seed_topics[word2id[word]] = t_id
                except: 
                    print('Was not able to find %s in %s' % (str(word), str(rng)))    

        # format the data as a document term matrix (required for guided LDA)
        dtm = bow2dtm(docs, dictionary)
        dtm = dtm.astype(int) 
        # train the guided lda models
        trained_models = mt.train_guidedLDAs(dtm, 
                                       mconfig['iters'], 
                                       mconfig['ntrange'], 
                                       seed_topics, 
                                       seed_confidence=mconfig['seed_confidence'])
        # print how long it took to train
        #print('Training all the models on the corpus for %s took %s' % (str(rng), str(time.time() - st)))
        #total_time += float(time.time() - st) # start counting the total time it takes for training. 
        name = 'G-LDA_'+rng+'_'+dconf+'_'+mconfig['name'] # Each model will be named after its data configuration and corpus range
        models_dir = os.getcwd() + '/models/'+name # Set the model directory name (will be created if does not exist)    
        if not os.path.isdir(models_dir): 
            os.mkdir(models_dir)
        try:
            bn.compressed_pickle(models_dir+'/'+name,trained_models)    
        except:
            pass
        # save the models to a dictionary 
        model_directory[rng] = {}
        model_directory[rng]['models'] = trained_models
        model_directory[rng]['dictionary'] = dictionary 
        model_directory[rng]['docs'] = docs
    #print(total_time)
    
    return model_directory
'''


def print_coherence_rankings(coherences, cm):
    avg_coherence = \
        [(num_topics, avg_coherence)
         for num_topics, (_, avg_coherence) in coherences.items()]
    ranked = sorted(avg_coherence, key=lambda tup: tup[1], reverse=True)
    print("Ranked by average '%s' coherence:\n" % cm.coherence)
    for item in ranked: 
        print("num_topics=%d:\t%.4f" % item)
    print("\nBest: %d" % ranked[0][0])
    
    return ranked



def determine_coherence(trained_models, dictionary, docs):

    # This performs a single pass over the reference corpus, accumulating
    # the necessary statistics for all of the models at once.
    cm = models.CoherenceModel.for_models(
        trained_models.values(), dictionary = dictionary, texts=docs, coherence='c_v')

    coherence_estimates = cm.compare_models(trained_models.values())

    coherences = dict(zip(trained_models.keys(), coherence_estimates))

    ranked = print_coherence_rankings(coherences, cm)
    
    return ranked, coherences 



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-# GET  LDA  RESULTS #-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#



def write_lda_descriptions(topic_des_path, model, num_words):
    num_words = int(num_words)
    with open(topic_des_path,'w') as f:
        for t in range(0,model.num_topics):
            f.write('\ntopic {} words: ,'.format(t) + ', '.join([w[0] for w in model.show_topic(t, num_words)]))
            f.write('\ntopic {} coefs: ,'.format(t) + ', '.join([str(w[1]) for w in model.show_topic(t, num_words)]))



def get_sentence_topics(model, corpus, fulldocs, path=None):
    sentence_topics_df = pd.DataFrame()
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # get the dominant topic, perc contribution and keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j==0: # dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sentence_topics_df=sentence_topics_df.append(pd.Series([int(topic_num), 
                                                                        round(prop_topic, 3),
                                                                        topic_keywords]), ignore_index=True)
            else: 
                break # no need to go past the dominant topic
    #sentence_topics_df.columns = ['Dominant_Topic','Perc_Contribution','Topic_Keywords']
    # add the original text to the output 
    contents = pd.Series(fulldocs)
    sentence_topics_df = pd.concat([sentence_topics_df, contents], axis=1)
    sentence_topics_df.columns = ['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    # if a path was submitted we save the data to a compressed pickle file 
    if path is not None:
        bn.compressed_pickle(path, sentence_topics_df)
            
    return sentence_topics_df
            
            













