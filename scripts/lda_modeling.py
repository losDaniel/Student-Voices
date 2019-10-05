import os, random, argparse, re, time
import pip._internal
import utils as bn 
import visuals as vs 
import lda_analysis as ld 
try:
    import numpy as np 
except:
    pip._internal.main(['install', 'pandas'])
    import numpy as np 

# set the seeds for random processes
random.seed(3)  
np.random.seed(3)


def lda_parameters_hardcodes(ranges):
    '''Set LDA parameters'''
    lda_parameters={}

    lda_parameters['LDA1']={} 
    for rng in ranges: 
        clen = len(range_indices[rng])
    
        lda_parameters['LDA1'][rng]={}
        lda_parameters['LDA1'][rng]['ntrange'] = list(range(10,40,5)) + list(range(40,101,15))
        lda_parameters['LDA1'][rng]['review_length'] = 100
        lda_parameters['LDA1'][rng]['passes'] = 40
        lda_parameters['LDA1'][rng]['nbelow'] = 30
        lda_parameters['LDA1'][rng]['nabove'] = .5
        lda_parameters['LDA1'][rng]['corpus_length']=clen   
        
    lda_parameters['LDA1']['[95, 101)']['review_length'] = 175 
    
    lda_parameters['LDA2']={} 
    for rng in ranges: 
        clen = len(range_indices[rng])
    
        lda_parameters['LDA2'][rng]={}
        lda_parameters['LDA2'][rng]['ntrange'] = list(range(10,40,5)) + list(range(40,101,15))
        lda_parameters['LDA2'][rng]['review_length'] = 150
        lda_parameters['LDA2'][rng]['passes'] = 40
        lda_parameters['LDA2'][rng]['nbelow'] = 30
        lda_parameters['LDA2'][rng]['nabove'] = .5
        lda_parameters['LDA2'][rng]['corpus_length']=clen   
        
    lda_parameters['LDA2']['[95, 101)']['review_length'] = 250 
    
    lda_parameters['LDA3']={} 
    for rng in ranges: 
        clen = len(range_indices[rng])
    
        lda_parameters['LDA3'][rng]={}
        lda_parameters['LDA3'][rng]['ntrange'] = list(range(10,40,5)) + list(range(40,101,15))
        lda_parameters['LDA3'][rng]['review_length'] = 100
        lda_parameters['LDA3'][rng]['passes'] = 40
        lda_parameters['LDA3'][rng]['nbelow'] = 50
        lda_parameters['LDA3'][rng]['nabove'] = .4
        lda_parameters['LDA3'][rng]['corpus_length']=clen   
        
    lda_parameters['LDA3']['[95, 101)']['review_length'] = 175 

    lda_parameters['LDA4']={} 
    for rng in ranges: 
        clen = len(range_indices[rng])
    
        lda_parameters['LDA4'][rng]={}
        lda_parameters['LDA4'][rng]['ntrange'] = list(range(10,40,5)) + list(range(40,101,15))
        lda_parameters['LDA4'][rng]['review_length'] = 150
        lda_parameters['LDA4'][rng]['passes'] = 40
        lda_parameters['LDA4'][rng]['nbelow'] = 50
        lda_parameters['LDA4'][rng]['nabove'] = .4
        lda_parameters['LDA4'][rng]['corpus_length']=clen   
        
    lda_parameters['LDA4']['[95, 101)']['review_length'] = 250 

    lda_parameters['LDA5']={} 
    for rng in ranges: 
        clen = len(range_indices[rng])
    
        lda_parameters['LDA5'][rng]={}
        lda_parameters['LDA5'][rng]['ntrange'] = list(range(10,40,5)) + list(range(40,101,15))
        lda_parameters['LDA5'][rng]['review_length'] = 125
        lda_parameters['LDA5'][rng]['passes'] = 40
        lda_parameters['LDA5'][rng]['nbelow'] = 30
        lda_parameters['LDA5'][rng]['nabove'] = .5
        lda_parameters['LDA5'][rng]['corpus_length']=clen   
        
    lda_parameters['LDA5']['[95, 101)']['review_length'] = 200 
    
    lda_parameters['LDA6']={} 
    for rng in ranges: 
        clen = len(range_indices[rng])
    
        lda_parameters['LDA6'][rng]={}
        lda_parameters['LDA6'][rng]['ntrange'] = list(range(10,40,5)) + list(range(40,101,15))
        lda_parameters['LDA6'][rng]['review_length'] = 175
        lda_parameters['LDA6'][rng]['passes'] = 40
        lda_parameters['LDA6'][rng]['nbelow'] = 30
        lda_parameters['LDA6'][rng]['nabove'] = .5
        lda_parameters['LDA6'][rng]['corpus_length']=clen   
        
    lda_parameters['LDA6']['[95, 101)']['review_length'] = 300 
    
    lda_parameters['LDA7']={} 
    for rng in ranges: 
        clen = len(range_indices[rng])
    
        lda_parameters['LDA7'][rng]={}
        lda_parameters['LDA7'][rng]['ntrange'] = list(range(10,40,5)) + list(range(40,101,15))
        lda_parameters['LDA7'][rng]['review_length'] = 125
        lda_parameters['LDA7'][rng]['passes'] = 40
        lda_parameters['LDA7'][rng]['nbelow'] = 50
        lda_parameters['LDA7'][rng]['nabove'] = .4
        lda_parameters['LDA7'][rng]['corpus_length']=clen   
        
    lda_parameters['LDA7']['[95, 101)']['review_length'] = 200 

    lda_parameters['LDA8']={} 
    for rng in ranges: 
        clen = len(range_indices[rng])
    
        lda_parameters['LDA8'][rng]={}
        lda_parameters['LDA8'][rng]['ntrange'] = list(range(10,40,5)) + list(range(40,101,15))
        lda_parameters['LDA8'][rng]['review_length'] = 175
        lda_parameters['LDA8'][rng]['passes'] = 40
        lda_parameters['LDA8'][rng]['nbelow'] = 50
        lda_parameters['LDA8'][rng]['nabove'] = .4
        lda_parameters['LDA8'][rng]['corpus_length']=clen   

    lda_parameters['LDA8']['[95, 101)']['review_length'] = 300

    return lda_parameters



def gen_lda_results(rng, setting, config, text, full_text, data, range_indices, lda_parameters, coherence_guide, models_dir, num_words=20):
    '''
    Generate the LDA Results: trained models, coherence scores, bubble graphs and topic assignment for documents
    '''
    
    run_name = setting+'_'+config                                                  # we will create results saving for each 
    name = setting+'_'+config+'_'+rng                
    
    indices = data.loc[range_indices[rng],'Review_Length']                                  # get the review length of the rows in range 
    filtered_index = indices[indices>lda_parameters[setting][rng]['review_length']].index   # filter it down by review length 
    print('Filtered the index for review length', flush=True)
    
    docs = [text[idx] for idx in filtered_index]                                            # get the documents that remain
    print('Retrieved the filtered documents, length is: '+str(len(docs)), flush=True)

    lda_parameters[setting][rng]['filtered_length'] = len(docs)                            # save the length of the results docs 
    lda_parameters[setting][rng]['setting'] = setting                                      # add the setting to the lda parameters for internal reference

    trained_models, corpus, dictionary, duration = ld.run_lda(docs,                        # train the lda models
                                                      lda_parameters[setting][rng],        # if all the LDA models have been trained this will just load them and return the other materials 
                                                      models_dir,
                                                      name)

    if 'duration' not in lda_parameters[setting][rng]: lda_parameters[setting][rng]['duration']=duration

    get_coherence=True
    if 'coherence_scores' in lda_parameters[setting][rng]:
        try:
            assert rng in coherence_guide
            assert config in coherence_guide[rng]
            assert setting in coherence_guide[rng][config]
            get_coherence=False            
            print('Coherence already estimated.', flush=True)            
        except:
            print('Coherence Guide and LDA parameters not in sync. Getting coherece...', flush=True)
    if get_coherence: 
        print('Determining coherence scores...')
        ranked, coherences = ld.determine_coherence(trained_models, dictionary, docs)          # get the ranked coherence models and the individual model's coherence scores 
        if rng not in coherence_guide: coherence_guide[rng]={}                                 # store the full coherence scores per topic in a separate dictionary object
        if config not in coherence_guide[rng]: coherence_guide[rng][config]={}                 
        if setting not in coherence_guide[rng][config]: coherence_guide[rng][config][setting]={}
    
        coherence_guide[rng][config][setting] = [(k,coherences[k]) for k in coherences]        # save results into the coherence guide  
        lda_parameters[setting][rng]['coherence_scores'] = ranked                              # save another copy to the LDA results 

        bn.full_pickle(os.getcwd()+'/results/coherence_scores_'+run_name, coherence_guide)
        print('Coherence Scores Saved.', flush=True)                           
        
        bn.full_pickle(os.getcwd()+'/results/lda_parameters_'+run_name, lda_parameters) 
        print('LDA Parameters Saved.', flush=True)

    fulldocs = [full_text[idx] for idx in filtered_index]                                  # get the full documents     

    best_topic_num = int(lda_parameters[setting][rng]['coherence_scores'][0][0])           # get the topic num with the top coherence score 
    model = trained_models[best_topic_num]                                                 # get the most coherent model 

    # specify paths to save the results  
    lda_viz_path = os.getcwd()+'/graphs/LDA Graphs/Viz_'+rng+'_'+str(best_topic_num)+'_'+setting+'_'+config+'.html'
    topic_des_path = os.getcwd()+'/results/LDA Descriptions/Des_'+rng+'_'+str(best_topic_num)+'_'+setting+'_'+config+'.csv'
    topic_vec_path = os.getcwd()+'/results/LDA Distributions/Vec_'+rng+'_'+str(best_topic_num)+'_'+setting+'_'+config # no extension because we will compress
    
    st=time.time()
    vs.save_topic_visualization(model, docs, dictionary, lda_viz_path)                     # create and save the topic pyLDAvis HTML topic visualization     
    print('Completed Visualizations in %s' % str(time.time()-st), flush=True)

    st=time.time()
    ld.write_lda_descriptions(topic_des_path, model, num_words)                            # save the top words for each topic and their coefficients 
    print('Completed Recording Key Words in %s' % str(time.time()-st), flush=True)

    st=time.time()
    ld.get_sentence_topics(model, corpus, fulldocs, path=topic_vec_path)                   # get main topic in each document
    print('Completed Recording Sentence Topics in %s' % str(time.time()-st), flush=True)

    return lda_parameters, coherence_guide



if __name__ == '__main__':
    
    # Home directory for the AWS instance
    os.chdir("/home/ec2-user/efs")

    parser = argparse.ArgumentParser(description='Launch spot instance')
    parser.add_argument('-c', '--configurations', help='Configuration A1,B1,C1,...', required=True)
    parser.add_argument('-s', '--settings', help='Settings LDA1,LDA2,...', required=True)
    args = parser.parse_args()

    print("Data configurations to work on are %s" % ', '.join(args.configurations.split(',')), flush=True)
    data_configurations = args.configurations.split(',')

    print("Settings to work on are %s" % ', '.join(args.settings.split(',')), flush=True)
    settings_to_run = args.settings.split(',')
    
    #~#~#~#~#~#~#~#~#~#~#~#~#~#
    #~#~# Load the Data #~#~#~#
    #~#~#~#~#~#~#~#~#~#~#~#~#~#
    
    print('Importing Rating Ranges...', flush=True)
    range_indices = bn.decompress_pickle(os.getcwd() + '/data/by_rating_range.pbz2')
    ranges = list(np.sort(list(range_indices.keys())))                                     # create a list of each range 

    print('Loading Statistics Data...', flush=True)
    data = bn.decompress_pickle(os.getcwd()+'/data/review_stats.pbz2')                     # load the review statistics dataset 

    print('Loading Full Text Data...', flush=True)
    full_text = bn.decompress_pickle(os.getcwd()+'/data/full_review_text.pbz2')            # load the full text so we can pull sample reviews 

    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    #~#~# Create  Directories #~#~#~#
    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    
    # check for an create the directories to store models and results
    if not os.path.exists(os.getcwd()+'/models/'):
        print('No "models" directory found. Creating...', flush=True)
        os.mkdir(os.getcwd()+'/models/')
        
    if not os.path.exists(os.getcwd()+'/graphs/'):
        print('No "graphs" directory found. Creating graphs & LDAGraphs...', flush=True)
        os.mkdir(os.getcwd()+'/graphs/')
        os.mkdir(os.getcwd()+'/graphs/LDAGraphs/')

    if not os.path.exists(os.getcwd()+'/results/LDAdescriptions/'):
        print('No "LDADescriptions" directory found. Creating...', flush=True)
        os.mkdir(os.getcwd()+'/results/LDAdescriptions/')

    if not os.path.exists(os.getcwd()+'/results/LDAdistributions/'):
        print('No "LDADistributions" directory found. Creating...', flush=True)
        os.mkdir(os.getcwd()+'/results/LDADistributions/')

    #~#~#~#~#~#~#~#~#~#~#
    #~#~# Modeling  #~#~#
    #~#~#~#~#~#~#~#~#~#~#

    print('Beginning modeling...', flush=True)
    for config in data_configurations:                                                     # Load each configuration 
        
        # load the cleaned text data
        print('Loading data...', flush=True)
        text, stem_map, lemma_map, phrase_frequencies = bn.decompress_pickle(os.getcwd()+'/data/cleaned_data/cleaned_docs_'+config+'.pbz2')

        for setting in settings_to_run:                                                    # Load each setting 
                                                                               
            model_directory = os.getcwd()+'/models/'+setting+'_'+config                    # Each config-setting will have its own model folder 
            if not os.path.exists(model_directory):                                        # Check if the folder exists for this config-setting pair 
                print('Model directory not detected. Creating...', flush=True) 
                os.mkdir(model_directory)                                                  # If it does not create it 
            else: 
                print('Model directory detected...', flush=True)                

            #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
            #~#~# Load Results Progress #~#~#~#
            #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
                                                                                           # Since config-setting pairs can be run on separate instances 
            run_name = setting+'_'+config                                                  # we will create results saving for each 

            # load the hardcoded lda parameters dictionaries, if a results updated version exists load that
            if os.path.exists(os.getcwd()+'/results/lda_parameters_'+run_name+'.pickle'):
                print('Existing log detected. Loading log...', flush=True)
                lda_parameters = bn.loosen(os.getcwd()+'/results/lda_parameters_'+run_name+'.pickle')
            else: 
                print('No Log Detected. Loading Hardcoded LDA Paramters...', flush=True)
                lda_parameters = lda_parameters_hardcodes(ranges)

            # load any coherence scores that have been registered 
            if os.path.exists(os.getcwd()+'/results/coherence_scores_'+run_name+'.pickle'):
                print('Existing coherence scores detected. Loading results...')
                coherence_guide = bn.loosen(os.getcwd()+'/results/coherence_scores_'+run_name+'.pickle')
            else: 
                coherence_guide = {} 

            # for each corpus (range) train a set of models with all the number of topics attempted in the ntrange key of the lda_parameters dictionary
            for rng in ranges: 

                print('Working on corpus '+str(rng), flush=True)
                # if there are coherence scores in the parameters dictionary the work has been completed and we can skip it 
                if 'coherence_scores' not in lda_parameters[setting][rng]:
                    
                    print(str(rng)+' '+str(setting)+' '+str(config), flush=True)
                    name = setting+'_'+config+'_'+rng                
                    lda_parameters[setting][rng]['data_configuration'] = config            # set the configuration for this lda_parameter setting and range
        
                    lda_parameters, coherence_guide = gen_lda_results(rng,                 # train models or load the ones that have already been trained 
                                                                      setting,
                                                                      config,
                                                                      text,
                                                                      full_text,
                                                                      data,
                                                                      range_indices,   
                                                                      lda_parameters,
                                                                      coherence_guide,
                                                                      model_directory)
         
                    print('Results have been saved. Analysis of '+name+' complete.', flush=True)

                    
                    
