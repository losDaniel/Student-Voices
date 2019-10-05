import utils as bn
import os
import time
import pandas as pd
import modeling_tools as mt
import multiprocessing
from collections import OrderedDict
from gensim import models

cpu_count = multiprocessing.cpu_count()-1
print("Will use %d of the available cores:" % cpu_count)

# Home directory for the AWS instance
os.chdir("/home/ec2-user/efs")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-# MODELING  FUNCTIONS #-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


# run lda models for a corpus given a set of parameters 
def run_lda(docs, params, models_dir, name):
    # keep track of time 
    st = time.time()
    # setup the text data with the specific model cleaning parameters 
    corpus, dictionary, literal_dictionary, id2word, word2id = mt.set_dictionary(docs, 
                                                                              nb=params['nbelow'], 
                                                                              na=params['nabove'])
    print('Set dictionary in %.3f seconds' % (time.time()-st), flush=True)
    # trains or loads pre-trained lda models 
    trained_models = train_ldas(corpus,
                                params['passes'],
                                params['ntrange'],
                                id2word,
                                models_dir,
                                name,
                                cores=cpu_count)
    # print how long it took to train
    duration = time.time() - st
    print('Training models for all the ranges on the corpus took %.3f' % duration, flush=True)
    
    return trained_models, corpus, dictionary, duration


# Train LDA topic models 
def train_ldas(docs, passes, ntrange, id2word, models_dir, name, cores = cpu_count):
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
    
    # we will train models for different numbers of topics and evaluate the coherence for each 
    for num_topics in ntrange: 

        model_path = os.path.join(models_dir, name + '%d.lda' % num_topics)

        if not os.path.exists(model_path): 
            print('Training '+name+' with '+str(num_topics)+' topics', flush=True)
            start = time.time()
            # train the model on multiple cores        
            lda = models.LdaMulticore(
                corpus = docs, id2word = id2word, num_topics = num_topics, 
                workers = cores, passes=passes, random_state = 2, eval_every = 1,
                alpha = 'asymmetric',
                decay=0.5, offset=64  # best params from Hoffman paper
            )         
            # save the model 
            lda.save(model_path, separately=False)

            # add it to the dictionary of trained models 
            trained_models[num_topics] = lda
            dur = time.time() - start
            print('Complete after '+str(dur)+' seconds', flush=True)

        else: 
            # load the saved model
            trained_models[num_topics] = models.LdaMulticore.load(model_path)
            print('model '+name+' with '+str(num_topics)+' topics loaded', flush=True)

    return trained_models



def save_models(named_models, models_dir, name):
    '''Model specifically to save models in dictionary of models'''
    if not os.path.isdir(models_dir): 
        os.mkdir(models_dir)
    for num_topics, model in named_models.items():
        model_path = os.path.join(models_dir, name + '%d.lda' % num_topics)
        model.save(model_path, separately=False)



def load_models(models_dir, name, ntrange):
    '''Load a dictionary of LDA models from a specific directory'''
    trained_models = OrderedDict()
    for num_topics in ntrange:
        model_path = os.path.join(models_dir, name + '%d.lda' % num_topics)
        #print("Loading LDA(k=%d) from %s" % (num_topics, model_path))
        trained_models[num_topics] = models.LdaMulticore.load(model_path)

    return trained_models
    


def print_coherence_rankings(coherences, cm):
    avg_coherence = \
        [(num_topics, avg_coherence)
         for num_topics, (_, avg_coherence) in coherences.items()]
    ranked = sorted(avg_coherence, key=lambda tup: tup[1], reverse=True)
    print("Ranked by average '%s' coherence:\n" % cm.coherence, flush=True)
    for item in ranked: 
        print("num_topics=%d:\t%.4f" % item, flush=True)
    print("\nBest: %d" % ranked[0][0], flush=True)
    
    return ranked



def determine_coherence(trained_models, dictionary, docs):
    '''Determine model coherence for a dictionary of trained models'''
    # This performs a single pass over the reference corpus, accumulating
    # the necessary statistics for all of the models at once.
    print('Estimating coherence for the models...', flush=True)
    start=time.time()
    cm = models.CoherenceModel.for_models(
        trained_models.values(), dictionary = dictionary, texts=docs, coherence='c_v')
    print('Finished in %s' % str(time.time()-start), flush=True)

    print('Comparing the models...', flush=True)
    start=time.time()
    coherence_estimates = cm.compare_models(trained_models.values())
    print('Finished in %s' % str(time.time()-start), flush=True)

    coherences = dict(zip(trained_models.keys(), coherence_estimates))

    ranked = print_coherence_rankings(coherences, cm)
    
    return ranked, coherences 



def write_lda_descriptions(topic_des_path, model, num_words):
    '''Write out the Key Words for each topic'''
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
    else: 
        return sentence_topics_df
            
            













