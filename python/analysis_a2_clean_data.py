import os, time, sys
import utils as bn 
import modeling_tools as mt 
import pip._internal
try:
    import pandas as pd 
except:
    pip._internal.main(['install', 'pandas'])
    import pandas as pd 


def gen_data():
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
    bn.compressed_pickle(os.getcwd() + '/data/review_stats', data)
        
    # we pickle and compress the text data
    bn.compressed_pickle(os.getcwd() + '/data/full_review_text', docs)
 
    

def create_ratings_bins(data):
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
    bn.compressed_pickle(os.getcwd() + '/data/by_rating_range', range_indices)

    

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

    # fifth configuration - FastText cleaning with Spell Check 
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



def clean_data(config, text, data_configurations):
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

    # track corpus length 
    cl2 = len(cleaned[0])
    # record the corpus lengths
    data_configurations[config]['length_prior'] = cl1
    data_configurations[config]['length_post'] = cl2
    
    # record the duration of the cleaning process
    data_configurations[config]['duration'] = time.time()-st
        
    bn.compressed_pickle(os.getcwd() + '/data/cleaned_data/cleaned_docs_'+str(config), cleaned)

    return data_configurations



if __name__ == '__main__':

    # Change the directory to the home directory. 
    os.chdir("/home/ec2-user/efs")

    sys.stdout.write('Beginning in '+os.getcwd()+'\n')
    # if the data is not present in the data directory create the datasets from the raw data   
    if not (os.path.exists(os.getcwd()+'/data/review_stats.pbz2') and os.path.exists(os.getcwd()+'/data/full_review_text.pbz2')):
        sys.stdout.write("Generating data..."+'\n')
        gen_data()
    sys.stdout.write('1')

    if not os.path.exists(os.getcwd()+'/data/cleaned_data/'):
        sys.stdout.write("Cleaned data directory not detected. Creating...")
        os.mkdir('data/cleaned_data/')
    sys.stdout.write('2')

    if not os.path.exists(os.getcwd()+'/data/by_rating_range.pbz2'):
        sys.stdout.write('File "by_rating_range.pbz2" not detected. Creating...')
        data = bn.decompress_pickle(os.getcwd()+'/data/review_stats.pbz2')
        # create a dictionary with the indices for each range
        create_ratings_bins(data)
    sys.stdout.write('3')

    # read the hard coded or saved data configuration settings 
    if not os.path.exists(os.getcwd()+'/results/cleaning_parameters.csv'):
        data_configurations = data_configuration_hardcodes()
    else: 
        sys.stdout.write('"cleaning_parameters.csv" detected. Loading...')
        data_configurations = pd.read_csv(os.getcwd()+'/results/cleaning_parameters.csv')
        data_configurations = data_configurations.set_index('Unnamed: 0').to_dict()
    sys.stdout.write('4')

    # check the cleaned documents in the data to check which have been completed 
    configurations = ['A1','B1','C1']
    to_clean = [] 
    for config in configurations: 
        if not os.path.exists(os.getcwd()+'/data/cleaned_data/cleaned_docs_'+config+'.pbz2'):
            to_clean.append(config)
    sys.stdout.write('Conifgurations: '+', '.join([c for c in to_clean]))
    sys.stdout.write('5')

    # we load the full corpus of review texts everytime we loop through a configuration 
    if len(to_clean)>0:
        sys.stdout.write('Loading "full_review_text.pbz2"')
        text = bn.decompress_pickle(os.getcwd() + '/data/full_review_text.pbz2')

    sys.stdout.write('6')
    # proceed with the ordinary cleaning steps
    for config in to_clean:
        sys.stdout.write('Cleaning '+str(config))
        data_configurations = clean_data(config, text, data_configurations)

    sys.stdout.write('7')
    if not os.path.exists(os.getcwd()+'/data/results/'):
        sys.stdout.write('Results folder not detected. Creating...')
        os.mkdir(os.getcwd()+'/data/results/')

    # record the cleaning process       
    experimental_setup = pd.DataFrame(data_configurations)
    experimental_setup.to_csv(os.getcwd()+'/results/cleaning_parameters.csv')       
    sys.stdout.write('Results summary saved to "cleaning_paramters.csv"')