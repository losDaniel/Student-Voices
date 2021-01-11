import os, sys
sys.path.append('../')

from student_voices import sv_utils as bn 
from student_voices import modeling_tools as mt 

import time, argparse
import pandas as pd 


def gen_data(save_dir):
    '''
    Import all the raw US data files from the data folder and compile them into 
    a text dataset and a statistics dataset then save these to the data folder. 
    '''    
    files = [f for f in list(os.walk(os.getcwd() + '/data'))[0][2] if 'US' in f]
    
    docs = []
    data = pd.DataFrame() 
    
    for f in files: 
    
        df = bn.loosen(os.getcwd() + '/data/' + f)
        
        # sometimes the website lists reviews where there are none and an empty file gets saved
        if len(df) == 0: # if we've loaded one of these empty files 
            continue # move on to the next
    
        df = df[[type(row['Text'])!=float for idx, row in df.iterrows()]]
            
        for doc in list(df['Text'].values):
            docs.append(doc)
    
        # if there was an unnames column (usually the index) we erase it 
        df = df.drop([c for c in df.columns if 'Unnamed' in c], 1)
        # we don't want the text but we want an idea, so we take the length 
        df['Review_Length'] = df['Text'].apply(lambda x: len(str(x)))
        # we drop the text but in this case since the data is saved by city we drop "State" and "City" and only save the filename
        df = df.drop(['Text','State','City'], 1)
        # we will be able to trace back specific reviews based on their file and index 
        df['File'] = str(f)
        df = df.reset_index().rename(columns = {'index':'FID'})
        
        # and append it to the main dataset 
        data = data.append(df, ignore_index = True, sort = False)
    
    # we pickle the data so it can be loaded very quickly 
    bn.compressed_pickle(save_dir + '/review_stats', data)
        
    # we pickle and compress the text data
    bn.compressed_pickle(save_dir + '/data/full_review_text', docs)
 
    
def create_hardcoded_ratings_bins(data, save_dir):
    '''
    Creates the hard-coded bins for ratings that created the corpus from the original study
    '''

    # normalize the ratings to fall in [0,100]
    data['Rating'] = (data['Rating']/data['Rating'].max()) * 100

    # we bin the data by ratings such that we end up with a similar number of rows in each rating bin 
    rating_bins = [0, 35, 60, 65, 75, 85, 95, 101]
        
    # split the ratings data into the bins above and place it in the Range field 
    data['Range'] = pd.cut(data['Rating'], rating_bins, right = False)
    range_dist = data['Range'].value_counts()
    
    # find the indices for each rating bin
    range_indices = {} 
    for v in range_dist.keys():
        range_indices[str(v)] = list(data.loc[data['Range']==v].index)
        
    # save a dictionary with the indices for each range under a key with that range
    bn.compressed_pickle(save_dir+'/by_rating_range', range_indices)
    

def data_configuration_hardcodes():
    '''Set data configurations'''
    data_configurations = {} 
    # first configuration - total cleaning 
    data_configurations['A1']={}
    data_configurations['A1']['lemmatizer']=True
    data_configurations['A1']['stemmer']=True
    data_configurations['A1']['remove_stops']=True
    data_configurations['A1']['no_not']=None
    data_configurations['A1']['remove_contractions']=True
    data_configurations['A1']['repeated_removal']=2
    data_configurations['A1']['gram']=4
    data_configurations['A1']['pthresh']=50
    data_configurations['A1']['spell_check']=False
        
    # second configuration - general cleaning without stemming
    data_configurations['B1']={}
    data_configurations['B1']['lemmatizer']=True
    data_configurations['B1']['stemmer']=False
    data_configurations['B1']['remove_stops']=True
    data_configurations['B1']['no_not']=None
    data_configurations['B1']['remove_contractions']=True
    data_configurations['B1']['repeated_removal']=2
    data_configurations['B1']['gram']=4
    data_configurations['B1']['pthresh']=50
    data_configurations['B1']['spell_check']=False
        
    # third configuration - general cleaning without stemming and with TextBlob spell check 
    data_configurations['C1']={}
    data_configurations['C1']['lemmatizer']=True
    data_configurations['C1']['stemmer']=False
    data_configurations['C1']['remove_stops']=True
    data_configurations['C1']['no_not']=None
    data_configurations['C1']['remove_contractions']=True
    data_configurations['C1']['repeated_removal']=2
    data_configurations['C1']['gram']=4
    data_configurations['C1']['pthresh']=50
    data_configurations['C1']['spell_check']=True
    
    # fourth configuration - Fasttext cleaning
    data_configurations['D1']={}
    data_configurations['D1']['lemmatizer']=False
    data_configurations['D1']['stemmer']=False
    data_configurations['D1']['remove_stops']=True
    data_configurations['D1']['no_not']=None
    data_configurations['D1']['remove_contractions']=True
    data_configurations['D1']['repeated_removal']=3 # fasttext interprets words as character n-grams so reductions may not be helpful
    data_configurations['D1']['gram']=None
    data_configurations['D1']['pthresh']=50 # irrelevant here because phrase finder is turned off 
    data_configurations['D1']['spell_check']=False

    # fifth configuration - stemming over lemming
    data_configurations['E1']={}
    data_configurations['E1']['lemmatizer']=False
    data_configurations['E1']['stemmer']=True
    data_configurations['E1']['remove_stops']=True
    data_configurations['E1']['no_not']=None
    data_configurations['E1']['remove_contractions']=True
    data_configurations['E1']['repeated_removal']=2
    data_configurations['E1']['gram']=4
    data_configurations['E1']['pthresh']=50
    data_configurations['E1']['spell_check']=False


    # sixth configuration - FastText cleaning with Spell Check 
    data_configurations['F1']={}
    data_configurations['F1']['lemmatizer']=False
    data_configurations['F1']['stemmer']=False
    data_configurations['F1']['remove_stops']=True
    data_configurations['F1']['no_not']=None
    data_configurations['F1']['remove_contractions']=True
    data_configurations['F1']['repeated_removal']=3 # fasttext interprets words as character n-grams so reductions may not be helpful
    data_configurations['F1']['gram']=None
    data_configurations['F1']['pthresh']=50 # irrelevant here becuase phrase finder is turned off
    data_configurations['F1']['spell_check']=True

    return data_configurations



def apply_data_cleaning_config(config, text, data_configurations, save_dir):
    # track time 
    st = time.time()
    # track corpus length 
    cl1 = len(text)
    # returns a tuple (cleaned documents, stem map, lemma map, phrase frequencies)
    cleaned= mt.pre_process(text,
                            lemmatizer=data_configurations[config]['lemmatizer'],
                            stemmer=data_configurations[config]['stemmer'],
                            remove_stops=data_configurations[config]['remove_stops'],
                            no_not=data_configurations[config]['no_not'], 
                            remove_contractions=data_configurations[config]['remove_contractions'],
                            repeated_removal=data_configurations[config]['repeated_removal'],
                            gram=data_configurations[config]['gram'],
                            pthresh=data_configurations[config]['pthresh'],
                            spell_check=data_configurations[config]['spell_check'])

    bn.compressed_pickle(save_dir+'/cleaned_docs_'+str(config), cleaned)

    print('cleaned_docs_'+str(config)+' has been saved...', flush=True)

    # track corpus length 
    cl2 = len(cleaned[0])
    # record the corpus lengths
    data_configurations[config]['length_prior'] = cl1
    data_configurations[config]['length_post'] = cl2
    
    # record the duration of the cleaning process
    data_configurations[config]['duration'] = time.time()-st
        
    return data_configurations



if __name__ == '__main__':

    # Home directory for the AWS instance
    os.chdir("/home/ec2-user/efs")

    parser = argparse.ArgumentParser(description='Launch spot instance')
    parser.add_argument('-c', '--configurations', help='Configuration A1,B1,C1,...', required=True)
    parser.add_argument('-p', '--path', help='The path to all the project data', required=True)
    parser.add_argument('-o', '--overwrite', help='Overwrite existing data', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.path+'/cleaned_data/'):
        os.mkdir(args.path+'/cleaned_data/')

#    if not os.path.exists(args.path+'/full_review_text.pbz2'):
#        raise Exception('full_review_text.pbz2 not found in --path, please use another path or create the file with clean_data.gen_data()')
#
#    if not os.path.exists(args.path+'/by_rating_range.pbz2'):
#        raise Exception('by_rating_range.pbz2 not found in --path, please use another path or create the file with clean_data.create_hardcoded_ratings_bins')

    data_configurations = data_configuration_hardcodes()

    # check the cleaned documents in the data to check which have been completed 
    configurations = args.configurations.split(',')
    to_clean = [] 
    for config in configurations: 
        if args.overwrite:
            to_clean.append(config)
        else: 
            if not os.path.exists(args.path+'/cleaned_docs_'+config+'.pbz2'):
                to_clean.append(config)
    print('Configurations: '+', '.join([c for c in to_clean]), flush=True)

    # we load the full corpus of review texts everytime we loop through a configuration 
    if len(to_clean)>0:
        print('Loading "full_review_text.pbz2"', flush=True)
        text = bn.decompress_pickle(args.path+'/full_review_text.pbz2')

    # proceed with the ordinary cleaning steps
    for config in to_clean:
        print('Cleaning '+str(config), flush=True)
        data_configurations = apply_data_cleaning_config(config, text, data_configurations, args.path+'/cleaned_data/')

    # record the cleaning process       
    experimental_setup = pd.DataFrame(data_configurations)
    experimental_setup.to_csv('/data/cleaning_parameters_'+'_'.join(configurations)+'.csv')       
    print('Results summary saved to "'+'/data/cleaning_parameters_'+'_'.join(configurations)+'.csv"', flush=True)