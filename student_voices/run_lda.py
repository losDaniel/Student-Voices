import os, sys
sys.path.append('../')

import argparse, time
import numpy as np
import pandas as pd 

import sv_utils as bn
import lda_analysis as ld

from path import Path 
root = Path(os.path.dirname(os.path.abspath(__file__)))


def run_lda_analysis(config, setting, model_dir, config_path): 

    st = time.time()
    
    text, stem_map, lemma_map, phrase_frequencies = bn.decompress_pickle(config_path+'/cleaned_docs_'+config+'.pbz2')
                    
    # define the directory where you want to save the models
    model_directory = model_dir+'/'+setting+'_'+config
    
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)

    # import the range indices 
    range_indices = bn.loosen(root + '/data/by_rating_range.pickle')
    # create a list of each range 
    ranges = list(np.sort(list(range_indices.keys())))
    # import hardcoded lda paramter dictionary 
    lda_parameters =ld.hardcoded_lda_parameters(ranges, range_indices)
    # import the data if need be
    data = bn.decompress_pickle(root+'/data/review_stats.pbz2')

    print('Starting LDA Analysis at '+str(time.time()-st))           
     
    # now for each range
    for rng in ranges: 
        print(rng,setting,config)
        # get the rows in the data that you need 
        indices = data.loc[range_indices[rng],'Review_Length']

        # filter the trainin corpus by review length and save the length 
        docs = [text[idx] for idx in indices[indices>lda_parameters[setting][rng]['review_length']].index]

        # save additional info to for the experimental record 
        lda_parameters[setting][rng]['filtered_length'] = len(docs)
        lda_parameters[setting][rng]['data_configuration'] = config
        lda_parameters[setting][rng]['setting'] = setting

        # train the lda models
        trained_models, corpus, dictionary, lda_parameters[setting][rng]['duration'] = ld.run_lda(docs, 
                                                                     lda_parameters[setting][rng])
        # name & save the lda models
        ld.save_models(trained_models, 
                       model_directory,
                       setting+'_'+config+'_'+rng)
        
        print('Finished '+setting+'_'+config+'_'+rng+' at '+str(time.time()-st))           
            
    # we save the outcomes of the data to the exprerimental setup dataset 
    experimental_setup = pd.DataFrame(lda_parameters[setting]).transpose()
    experimental_setup.sort_index().to_csv(model_directory+'/lda_register'+setting+'_'+config+'.csv')


if __name__ == '__main__':

    # Home directory for the AWS instance
    os.chdir("/home/ec2-user/efs")

    parser = argparse.ArgumentParser(description='Launch spot instance')
    parser.add_argument('-c', '--configuration', help='Configuration (A1,B1,C1,...)', required=True)
    parser.add_argument('-cp', '--configpath', help='Path to configuration data', required=True)
    parser.add_argument('-md', '--modeldir', help='Path to save the models', required=True)
    parser.add_argument('-s', '--setting', help='LDA parameter setting to use from hardcoded options', required=True)

    args = parser.parse_args()

    config = args.configuration
    config_path = args.configpath
    model_dir = args.modeldir
    setting = args.setting
    
    run_lda_analysis(config, setting, model_dir, config_path)


