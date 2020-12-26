import sys
sys.path.append('../')

import os, re, multiprocessing, string, time

import numpy as np
from student_voices import sv_utils as bn 

#import pip._internal
#try:
#    import pandas as pd 
#except:
#    pip._internal.main(['install', 'pandas'])
#    import pandas as pd 

#try:
#    from nltk.stem.wordnet import WordNetLemmatizer
#except:
#    pip._internal.main(['install', 'nltk'])
#    from nltk.stem.wordnet import WordNetLemmatizer
    
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

#try:
#    from gensim.models.phrases import Phrases, Phraser
#except:
#    pip._internal.main(['install', 'gensim'])
#    from gensim.models.phrases import Phrases, Phraser
from gensim.utils import save_as_line_sentence
from gensim.corpora import Dictionary

#try: 
#    from sklearn.feature_extraction.text import TfidfVectorizer
#except: 
#    pip._internal.main(['install','sklearn'])
#    from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC 

#try: 
#    from textblob import TextBlob
#except: 
#    pip._internal.main(['install','TextBlob']) # try again just to secure
#    pip._internal.main(['install','TextBlob'])
#    from textblob import TextBlob


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# 1. TEXT CLEANING METHODS  #-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# 2. SAVE TEXT AS NEWLINE FOR GENSIM  #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# 3. MODELING FUNCTIONS   #-#-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


cpu_count = multiprocessing.cpu_count()

# rootpath for the AWS directory 
os.chdir("/home/ec2-user/efs")
    
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# 1. TEXT CLEANING METHODS  #-#-#-#-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


# import contractions dictionary 
contractions = bn.loosen(os.getcwd()+'/data/contractions.pickle')
cnt = {} # we need to create a version the contractions that is compatible with the cleaning step (where apostrophes and lower case has been implemented)
for c in contractions: 
    w = c.lower().replace("'","")
    cnt[w] = contractions[c].lower()
contractions = cnt 
del cnt


# import stop words 
stops = set(stopwords.words('english'))
stops = [s.replace("'",'') for s in stops]
# pull in other stop words that have been added manually
for w in open(os.getcwd()+'/data/stop_words.txt','r').read().replace('\n','').replace("'","").split(','):
    if w not in stops: stops.append(w.strip())
   
     
# Main pre-processing function  
def pre_process(text, # a list of texts 
                lemmatizer=True, # do you want to lemmatize the text 
                stemmer=True, # do you want to stem the text 
                remove_stops=True, # do you want to remove stops 
                no_not= None, # list of stop words you do not want to remove 
                remove_contractions=True, # do you want to replace contractions 
                repeated_removal=2, # remove characters that repeat at least this number of times, set to None to cancel
                gram = 4, # None if you do not want to find phrases, find phrases of up to this number of words
                pthresh = 20, # how many times must a phrase appear to be included
                spell_check = False, # set to True if you want to use TextBlob to correct misspellings
                threads = None # set to some integer equal to the number of cores you want to use 
                ):
    # omit the threading process, no signficant speed-up & unrealiable doc ordering
    threads = None 
    if threads is not None: 
        print('ommitted')
        '''
        # parse the text equal chunks for each processor
        text_segments = bn.chunkify(text, threads)
        
        args = [(text,
                 lemmatizer,
                 stemmer,
                 remove_stops,
                 no_not,
                 remove_contractions,
                 repeated_removal,
                 spell_check) for text in text_segments]
        for a in args:
            print(len(a))
        docs, stem_map, lemma_map = thread_cleaning(args, threads)
        '''

    else: 
        # default stemmer is nltk's PorterStemmer, but other stemmers or lematizers can be submitted
        docs, stem_map, lemma_map = clean_docs(text, 
                                               lemmatizer=lemmatizer, 
                                               stemmer=stemmer, 
                                               remove_stops=remove_stops, 
                                               no_not=no_not,
                                               remove_contractions=remove_contractions, 
                                               repeated_removal=repeated_removal,
                                               spell_check=spell_check) 
        
    # we find bigrams and return a dictionary of phrase frequency as well 
    if gram is None: 
        phrase_dict = {}
    else: 
        docs, phrase_dict = find_phrases(docs, 
                                         phrase_thresh=pthresh, 
                                         gram=gram)

    return docs, stem_map, lemma_map, phrase_dict

# first cleaning function 
def clean_docs(docs, # list of text documents (not tokenized)
               lemmatizer=True, # lemmatize documents 
               stemmer=True, # stem documents 
               remove_stops=True, # remove stop words 
               no_not=None, # list of stop words you do not want to remove 
               remove_contractions=True, # do you want to replace contractions 
               repeated_removal=2, # remove characters that repeat at least this number of times 
               spell_check = False # set to True if you want to use TextBlob to correct misspellings
               ):
    '''Clean the documents and return the cleaned documents, a map of stemmed words, and a map of phrases their frequency.'''        
    
    print('Begnning Doc-wise Cleaning...', flush=True)

    # Clean each document 
    docs = docwise_cleaning(docs, repeated_removal=repeated_removal, remove_contractions=remove_contractions)

    print('Basic Cleaning: Complete', flush=True) 
   
    # Remove numbers, but not words that contain numbers. for the rmt corpus this is very useful in separating "grade" from "10th grade", "9th grade", ... which have very different meanings
    docs = rnumeric(docs)
    print('Filtering Out Numerics: Complete', flush=True)

    # Remove words that are only one character. 
    docs = [[token for token in doc if len(token) > 1] for doc in docs] 
    print('Remove One Character Words: Complete', flush=True)
    
    # remove stop words 
    if remove_stops: 
        stop_words = stops.copy()
        # remove your exceptions from the stopword list 
        if no_not is not None: 
            for n in no_not:
                try:
                    stop_words.remove(n)
                except:
                    pass
        docs = [[token for token in doc if token not in stop_words] for doc in docs] 
        print('Stop Word Removal: Complete', flush=True)
    
    # we can use the textblob module to implement spell_check and auto-corrections (this is even capable of capturing slang)
    if spell_check: 
        corrections = {} # TextBlob takes a bit to run so to avoid repeating lookups we create a dictionary 
        for idx in range(len(docs)): # not sure why I do this instead of just doing for doc in docs :/
            for token in docs[idx]: 
                if token not in corrections: 
                    corrections[token] = str(TextBlob(token).correct()) 
        # when textblob fails to recognize a word it returns the same word
        docs = [[corrections[token] for token in doc] for doc in docs]            
        print('Spell Check: Complete', flush=True)

    # we're going to stem the words but create a map so we can trace the words back. We do so by using the populate stems function defined above 
    lemma_map = {}
    if lemmatizer: 
        docs = [[populate_stems(token, lemma_map, stemmer = WordNetLemmatizer()) for token in doc] for doc in docs]
        print('Lemmatization: Complete', flush=True)

    stem_map = {}
    if stemmer: 
        docs = [[populate_stems(token, stem_map, stemmer = PorterStemmer()) for token in doc] for doc in docs]
        print('Stemming: Complete', flush=True)
        
    return docs, stem_map, lemma_map
    

def docwise_cleaning(docs, repeated_removal=None, remove_contractions=False):
    '''Apply basic cleaning to the documents in the corpus'''
    # instantiate a tokenizer 
    tokenizer = RegexpTokenizer(r'\w+')
    
    st = time.time()
    pt = time.time()

    for idx in range(0,len(docs)):
        try: 
            # remove unicode spacing characters 
            docs[idx] = docs[idx].replace('\r',' ')
            docs[idx] = docs[idx].replace('\n',' ')

            # convert to lower case 
            docs[idx] = docs[idx].lower()  
            
            # even after isolating the text variable I end up with the values from the SubmittedBy column inserted into the text. 
            docs[idx] = docs[idx].replace('submitted by a student','')
            docs[idx] = docs[idx].replace('submitted by a parent','')

            # remove punctuation
            docs[idx] = docs[idx].replace("'","") # remove apostrophes 
            docs[idx] = docs[idx].replace('[{}]'.format(string.punctuation), ' ')

            if repeated_removal is not None: 
                # replace `reapeated_removal` OR MORE repeated characters with a single instance (replace "coooool!!!!" with "col!")
                docs[idx] = re.sub(r'(.)\1{'+str(int(repeated_removal)-1)+',}', r'\1', docs[idx])

            # replace contractions (since we removed punctuation we replace versions of the contractions without apostrophes)
            if remove_contractions:
                for c in contractions: 
                    docs[idx] = re.sub("(?<![a-zA-Z])"+c.replace("'","")+"(?![a-zA-Z])",contractions[c], docs[idx])

            # replace some common typos in this corpus 
            docs[idx] = re.sub('foward','forward', docs[idx])
            docs[idx] = re.sub('yrs','years', docs[idx])
            # split into words 
            docs[idx] = tokenizer.tokenize(docs[idx])  
            if np.mod(idx,100000)==0:
                print(str((idx/len(docs))*100)+'%'+' Time: '+str(time.time()-st)+' Rate: '+str((time.time()-pt)), flush=True)
                pt = time.time()
        except Exception as e: 
            print('IDX: ', idx, docs[idx], flush=True)
            raise e

    return docs



def rnumeric(docs):
    '''Remove stand alone numeric characters'''
    st = time.time()
    pt = time.time()
    out = [] 
    for idx, doc in enumerate(docs): 
        mod_doc=[]
        if np.mod(idx,500000)==0:
            print(str((idx/len(docs))*100)+'%'+' Time: '+str(time.time()-st)+' Rate: '+str((time.time()-pt)), flush=True)
            pt = time.time()
        for token in doc:
            if not token.isnumeric():
                mod_doc.append(token)
        out.append(mod_doc)
    return out 



# Creating and indexing stems or lemmatizations 
def populate_stems(word, # the word you want to stem or lemmatize
                   stem_map, # a stem frequency map/dictionary 
                   stemmer = PorterStemmer()): # default stemmer 
    '''A function to stem words while keeping track of the original words the stems are tied to'''
    # instantiate the stemmer or lemmatizer
    p_stemmer = stemmer
    
    # get the word stem
    if 'Lemma' in str(stemmer):
        tok = p_stemmer.lemmatize(word) 
    
    elif 'Stem' in str(stemmer): 
        tok = p_stemmer.stem(word)
    
    else: 
        raise Exception('Must use NLTK Stemmer or Lematizer')
    
    # if the stem has been mapped 
    if tok in stem_map.keys(): 
        
        # check if this version of the word has already been mapped to the stem
        if word in stem_map[tok].keys():
            stem_map[tok][word] += 1 # if it has tally this instance 
        
        else: # if it has not 
            stem_map[tok][word] = 1 # add an entry for this version of the word to the register dic for this stem
    
    else: # if the stem is not in the map
        stem_map[tok] = {} # create a dic to register the frequency of the words associated with this stem
        stem_map[tok][word] = 1 
        
    return tok


# Identify and return phrases 
def find_phrases(docs, phrase_thresh = 40, gram = 2):
    '''
    Use gensim to replace common ngrams with phrases. 
    - docs : list, list of documents to get X-grams for 
    - phrase_thresh : int, threshold for number of times an X-gram must appear
    - gram : int, default is 2 for bigram. 3 gets trigrams and so on.  
    '''
    # we keep a dictionary for phrase frequency
    phrase_voc = {}
    if gram < 2: 
        raise Exception("That don't make no sense, gram should be > 2")
    # we start with bigrams
    g = 2 
    st = time.time()
    while g <= gram:     
        phrases = Phrases(docs, min_count=phrase_thresh)  # train model 
        for phrase, score in phrases.export_phrases(docs): 
            if phrase not in phrase_voc: 
                phrase_voc[phrase] = score 
        phrases = Phraser(phrases) 
        docs = [phrases[doc] for doc in docs] 
        # once we've embedded the bigrams in the docs we go for trigrams and so on 
        print('Finished searching for grams '+str(g)+' after '+str(time.time()-st))
        g += 1 

    return docs, phrase_voc



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-# 2. SAVE TEXT AS NEWLINE FOR GENSIM  #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


# Save training data as line by line text files, speeds up gensim functions considerably
def setup_filebase(docs, none_below, not_above, filename, overwrite=True):
    '''Sets up line by line text files for Gensim training'''

    # load in the documents (cleaned, )
    data, dictionary, literal_dictionary, id2word, word2id = setup_text_training_data(docs, none_below, not_above)

    # save the corpus in the .txt format needed for filebased training  
    txtfile = os.getcwd() + '/data/cleaned_data/vectrain_'+filename+'.txt'
    if overwrite: # create or overwrite the text file 
        save_as_line_sentence(data, txtfile)    
    elif not os.path.exists(txtfile):
        save_as_line_sentence(data, txtfile)  

    return txtfile, dictionary


# clean the corpus and return training data (This is necessary when filtering a corpus will leave you with empty documents, this is unlikely in cases where review length is controlled) 
def setup_text_training_data(docs, none_below, not_above):

    # populate the dictionary elements and filter out low and high occurring words
    corpus, dictionary, literal_dictionary, id2word, word2id = set_dictionary(docs, nb=none_below, na=not_above)
    # use the literal dictionary to filter the docs with the dictionary created in gensim
    data = filter_corpus(docs, literal_dictionary) 

    return data, dictionary, literal_dictionary, id2word, word2id


# Setup dictionaries 
def set_dictionary(docs, nb=30, na=0.5):
    print(docs[0], flush=True)
    dictionary = Dictionary(docs)
    print('Length of the dictionary is %s' % str(len(dictionary)), flush=True)

    # filter out words that occur less than X documents, or more than Y% of the documents.
    dictionary.filter_extremes(no_below=nb, no_above=na)

    # vectorize the docs by creating bag-of-words representations of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Done filtering (in)frequent words', flush=True) 
    print('Length of the corpus is %s' % str(len(corpus)), flush=True)
    print('Length of the dictionary is %s' % str(len(dictionary)), flush=True)

    # make an index to word dictionary to feed to the id2word argumend in the lda training command. 
    temp = dictionary[list(dictionary.keys())[0]]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    # We also need a word to id dictionary for the guided LDA 
    word2id = {} 
    for ids in id2word: 
        word2id[id2word[ids]] = ids

    # create a list with all the words in the finalized dictionary
    literal_dictionary = [dictionary[i] for i in dictionary]

    return corpus, dictionary, literal_dictionary, id2word, word2id 


# Clean corpus to match dictionary filters 
def filter_corpus(docs, lit_dict): # do not apply if using stanford NLP Parser
    '''Only keep the words in the dictionary'''
    out_data = []
    for doc in docs: 
    	out_data.append([word for word in doc if word in lit_dict])
    out_data = [doc for doc in out_data if len(doc) > 0]

    return out_data



#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#-#-# MODELING  FUNCTIONS #-#-#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#


# TFIDF transform and run an SVM on a text
def run_svm(docs, labels):

    # format training and testing datasets for tfidf    
    tfidf_vec = TfidfVectorizer()
    # we feed it sentences it can split 
    X_train = tfidf_vec.fit_transform([' '.join(doc) for doc in docs]) 
    y_train = labels
    
    svm = LinearSVC()
    svm.fit(X_train, y_train)

    feature_names = tfidf_vec.get_feature_names()
    coef = svm.coef_.ravel()

    results = pd.DataFrame({"coef":coef,"word":feature_names})

    return results, svm, tfidf_vec



def extract_seed(seed_list):
    seeds = {} 
    for f in seed_list: 
        seed_name = f.split('\\')[-1].replace('.txt','')
        with open(f, 'r') as s: 
            seed_words = list(filter(None,s.read().split('\n')))
        seeds[seed_name] = seed_words 
        
    return seeds



def search_corpus(word, docs):
    indices = []
    frequencies = []
    for i,c in enumerate(docs):
        # if the word is in the document
        if word in c:
            indices.append(i)
            frequencies.append(len([w for w in c if w==word]))
    return indices, frequencies 



def build_freq_dict(docs, full_seeds, primary_seeds):
    freq_dict = {}
    for k in full_seeds: 
        for word in full_seeds[k]:
            if word not in freq_dict: 
                freq_dict[word] = {}

            # get the indices of the docs where the word can be found and its frequency in each doc 
            freq_dict[word]['indices'], freq_dict[word]['frequencies'] = search_corpus(word, docs)

            # attach the seed to each word 
            if 'seeds' not in freq_dict[word]:
                freq_dict[word]['seeds'] = []
            freq_dict[word]['seeds'].append(k)

            # and whether or not it is primary for the current seed 
            if 'primary' not in freq_dict[word]:
                freq_dict[word]['primary'] = {}
            freq_dict[word]['primary'][k] = (word in primary_seeds[k+'_primary'])
    
    return freq_dict 



def seed_frequencies(corpus, full_seeds, freq_dict):
    dataset = {} 

    for seed in full_seeds: 
        for word in freq_dict: 
            if seed in freq_dict[word]['seeds']:
                # make space for the seed information if not yet 
                if seed not in dataset:
                    dataset[seed] = {}
                # make space for the indices if not yet 
                if 'indices' not in dataset[seed]:
                    dataset[seed]['indices'] = []
                # append the indices for the new word
                dataset[seed]['indices'] += freq_dict[word]['indices']
                # get the unique documents (no need to double count)
                dataset[seed]['indices'] = list(set(dataset[seed]['indices']))


        dataset[seed]['n_occurence'] = len(dataset[seed]['indices'])
        dataset[seed]['freq'] = len(dataset[seed]['indices'])/len(corpus)

    dataframe = pd.DataFrame({'seeds':list(dataset.keys()),
                              'occurrence':[dataset[s]['n_occurence'] for s in dataset],
                              'freq':[dataset[s]['freq'] for s in dataset]})

    return dataframe, dataset



def destem_list(stem_map, word_list):
    raw_list = []
    for word in word_list:
        if '_' in word: 
            new_wrd = '_'.join([[e for e in stem_map[w] if stem_map[w][e]==max(stem_map[w].values())][0] for w in word.split('_')])
        else: 
            new_wrd = [e for e in stem_map[word] if stem_map[word][e]==max(stem_map[word].values())][0]
        raw_list.append(new_wrd)
    return raw_list



def seed_descriptions(full_seeds, stem_map):
    mlen = max([len(full_seeds[seed]) for seed in full_seeds])
    filled_seeds = {}
    for seed in full_seeds: 
        
        filled_seeds[seed] = set(list(full_seeds[seed].copy()))
        filled_seeds[seed] = destem_list(stem_map, filled_seeds[seed])
        
        while len(filled_seeds[seed]) < mlen: 
            filled_seeds[seed].append('')
        
    df = pd.DataFrame(filled_seeds)
            
    return df 