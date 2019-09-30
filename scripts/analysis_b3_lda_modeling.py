import os, random, argparse
import pip._internal
import utils as bn 
import lda_analysis as ld
try:
    import pandas as pd 
except:
    pip._internal.main(['install', 'pandas'])
    import pandas as pd 
try:
    import numpy as np 
except:
    pip._internal.main(['install', 'pandas'])
    import numpy as np 

# set the seeds for random processes
random.seed(3)  
np.random.seed(3)

# Change the directory to the home directory. 
os.chdir('..')

def gen_lda_results(rng, setting, text, data, range_indices, lda_parameters, coherence_guide, models_dir, name):
    # get the rows in the data that you need 
    indices = data.loc[range_indices[rng],'Review_Length']

    filtered_index = indices[indices>lda_parameters[setting][rng]['review_length']].index
    print('Filtered the index for review length', flush=True)
    
    docs = [text[idx] for idx in filtered_index]
    print('Retrieved the filtered documents, length is:', flush=True)
    print(len(docs), flush=True)
    # save additional info to for the experimental record 
    lda_parameters[setting][rng]['filtered_length'] = len(docs)
    lda_parameters[setting][rng]['setting'] = setting

    # if all the LDA models have been trained this will just load them and return the other materials 
    trained_models, corpus, dictionary, duration = ld.run_lda(docs,
                                                      lda_parameters[setting][rng],
                                                      models_dir,
                                                      name)

    if 'duration' not in lda_parameters[setting][rng]: lda_parameters[setting][rng]['duration']

    print('Determining coherence scores...')
    # get the ranked coherence models and the coherence
    ranked, coherences = ld.determine_coherence(trained_models, dictionary, docs)

    # store the full coherence scores per topic in a separate dictionary object
    if rng not in coherence_guide: coherence_guide[rng]={} 
    if config not in coherence_guide[rng]: coherence_guide[rng][config]={}
    if setting not in coherence_guide[rng][config]: coherence_guide[rng][config][setting]={}

    coherence_guide[rng][config][setting] = [(k,coherences[k]) for k in coherences]

    # save the coherence scores to the experimental register
    lda_parameters[setting][rng]['coherence_scores'] = ranked

    return trained_models, corpus, dictionary, lda_parameters, coherence_guide



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



if __name__ == '__main__':
    
    # Home directory for the AWS instance
    os.chdir("/home/ec2-user/efs")

    parser = argparse.ArgumentParser(description='Launch spot instance')
    parser.add_argument('-c', '--configurations', help='Configuration A1,B1,C1,...', required=True)
    args = parser.parse_args()
    data_configurations = args.configurations.split(',')
    cname = '_'.join(data_configurations)
    
    #~#~#~#~#~#~#~#~#~#~#~#~#~#
    #~#~# Load the Data #~#~#~#
    #~#~#~#~#~#~#~#~#~#~#~#~#~#
    
    print('Importing Rating Ranges...', flush=True)
    range_indices = bn.decompress_pickle(os.getcwd() + '/data/by_rating_range.pbz2')
    ranges = list(np.sort(list(range_indices.keys())))                                 # create a list of each range 

    print('Loading Statistics Data...', flush=True)
    data = bn.decompress_pickle(os.getcwd()+'/data/review_stats.pbz2')                 # load the review statistics dataset 

    print('Loading Full Text Data...', flush=True)
    full_text = bn.decompress_pickle(os.getcwd()+'/data/full_review_text.pbz2')        # load the full text so we can pull samples 

    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    #~#~# Load Results Progress #~#~#~#
    #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

    # load the hardcoded lda parameters dictionaries, if a results updated version exists load that
    if os.path.exists(os.getcwd()+'/results/lda_parameters_'+cname+'.pickle'):
        print('Existing log detected. Loading log...', flush=True)
        lda_parameters = bn.loosen(os.getcwd()+'/results/lda_parameters_'+cname+'.pickle')
    else: 
        print('No Log Detected. Loading Hardcoded LDA Paramters...', flush=True)
        lda_parameters = lda_parameters_hardcodes(ranges)

    # load any coherence scores that have been registered 
    if os.path.exists(os.getcwd()+'/results/coherence_scores_'+cname+'.pickle'):
        print('Existing coherence scores detected. Loading results...')
        coherence_guide = bn.loosen(os.getcwd()+'/results/coherence_scores_'+cname+'.pickle')
    else: 
        coherence_guide = {} 

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

    if not os.path.exists(os.getcwd()+'/results/LDADescriptions/'):
        print('No "LDADescriptions" directory found. Creating...', flush=True)
        os.mkdir(os.getcwd()+'/results/LDADescriptions/')

    if not os.path.exists(os.getcwd()+'/results/LDADistributions/'):
        print('No "LDADistributions" directory found. Creating...', flush=True)
        os.mkdir(os.getcwd()+'/results/LDADistributions/')

    #~#~#~#~#~#~#~#~#~#~#
    #~#~# Modeling  #~#~#
    #~#~#~#~#~#~#~#~#~#~#

    print('Beginning modeling...', flush=True)
    for config in data_configurations:
        
        # load the cleaned text data
        print('Loading data...', flush=True)
        text, stem_map, lemma_map, phrase_frequencies = bn.decompress_pickle(os.getcwd()+'/data/cleaned_data/cleaned_docs_'+config+'.pbz2')

        for setting in lda_parameters:
            # check if the models directory for the current configuration exists 
            model_directory = os.getcwd()+'/models/'+setting+'_'+config

            if not os.path.exists(model_directory):
                print('Model directory not detected. Creating...', flush=True)
                os.mkdir(model_directory)
            else: 
                print('Model directory detected...', flush=True)                

            # for each corpus (range) train a set of models with all the number of topics attempted in the ntrange key of the lda_parameters dictionary
            for rng in ranges: 
                print('Working on corpus '+str(rng), flush=True)
                # if there are coherence scores in the parameters dictionary the work has been completed and we can skip it 
                if 'coherence_scores' not in lda_parameters[setting][rng]:
                    
                    print(str(rng)+' '+str(setting)+' '+str(config), flush=True)
                    
                    name = setting+'_'+config+'_'+rng                
                    # set the configuration for this lda_parameter setting and range
                    lda_parameters[setting][rng]['data_configuration'] = config
    
                    # train models or load the ones that have already been trained 
                    trained_models,corpus,dictionary,lda_parameters,coherence_guide=gen_lda_results(rng,
                                                                                                    setting,
                                                                                                    text,
                                                                                                    data,
                                                                                                    range_indices,   
                                                                                                    lda_parameters,
                                                                                                    coherence_guide,
                                                                                                    model_directory, 
                                                                                                    name)
                    # save any progress on coherence scores
                    bn.full_pickle(os.getcwd()+'/results/coherence_scores_'+cname, coherence_guide)
                    print('Coherence Scores Saved.', flush=True)
                    
                    # save any progress on the lda parameters
                    bn.full_pickle(os.getcwd()+'/results/lda_parameters_'+cname, lda_parameters)
                    print('LDA Parameters Saved.', flush=True)

    print('Finished for all configs...', flush=True)
    experimental_setup = pd.DataFrame()
    for setting in lda_parameters: 
            # we save the outcomes of the lda training to the exprerimental setup dataset 
            experimental_setup = experimental_setup.append(pd.DataFrame(lda_parameters[setting]).transpose())

    experimental_setup.to_csv(os.getcwd()+'/results/experimental_register'+cname+'.csv')
    print('Saved experimental register...', flush=True)