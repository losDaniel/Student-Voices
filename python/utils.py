import bz2
import _pickle as cPickle
import pickle 

def chunks(l,n):
    '''Break list l up into chunks of size n'''    
    for i in range(0, len(l), n):
        yield l[i:i+n]  

def chunkify(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def order(frame, var):
    '''Brings the variables supplied to the front of the dataframe. 
    
    - frame : pandas dataframe
    - var : str or list of str. Variables you want to bring to the front. 
    '''
    if type(var) == str:
        var = [var]          
    varlist =[w for w in frame.columns if w not in var]
    frame = frame[var+varlist]
    return frame   

def full_pickle(title, data):
    '''pickles the submited data and titles it
    '''
    pikd = open(title + '.pickle', 'wb')
    pickle.dump(data, pikd)
    pikd.close()   
    
def loosen(file):
    '''loads and returns a pickled objects
    '''
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data   

def compressed_pickle(title, data):
    '''
    Pickle a file and then compress it into a file with extension .pbz2
    __________
    parameters
    - title : title of the file you want to save (will be saved with .pbz2 extension automatically)
    - data : object you want to save 
    '''
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(filename):
    '''filename - file name including .pbz2 extension'''
    data = bz2.BZ2File(filename, 'rb')
    data = cPickle.load(data)
    return data

def rename(data, oldnames, newname): 
    if type(oldnames) == str:
        oldnames = [oldnames]
        newname = [newname]
    i = 0 
    for name in oldnames:
        oldvar = [c for c in data.columns if name in c]
        if len(oldvar) == 0: 
            raise ValueError("Sorry, couldn't find that column in the dataset")
        if len(oldvar) > 1: 
            print("Found multiple columns that matched " + str(name) + " :")
            for c in oldvar:
                print(str(oldvar.index(c)) + ": " + str(c))
            ind = input('please enter the index of the column you would like to rename: ')
            oldvar = oldvar[int(ind)]
        if len(oldvar) == 1:
            oldvar = oldvar[0]
        data = data.rename(columns = {oldvar : newname[i]})
        i += 1 
    return data    

def firstrow_header(df):
    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    df.columns = new_header #set the header row as the df header
    return df


class notes:
    def init(self,file):
        self.file = file 

    def write(self, text):
        if os.path.exists(self.file):
            with open(self.file, 'a') as f: 
                f.write(text+'\n')
                f.close() 
        else:
            with open(self.file,'w') as f: 
                f.write(text+'\n')
                f.close()

