"""
@author: Carlos Valcarcel

ratemyteacher.com webscraping methods. Contains the methods to scrape the hierarchically
ordered pages on ratemyteacher.com. 
"""

import os, re, time
import urllib.request as urllib
import pandas as pd
from bs4 import BeautifulSoup
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
import bear_necessities as bn


class rmt_scraper:
    

    def __init__(self, url): 

        self.home_url = ''
        self.home_url = url 
        print('Scraper setup for %s' % self.home_url)

        self.states = [] 
        self.states = get_states(self.home_url)   
        print('Populating states')
        
        self.site_map = pd.DataFrame()
        self.school_database = pd.DataFrame()
        
        

    def check_for_tups(self, lot):
        '''Error catching for lists that are not formatted correctly'''
        if (len(list(set([type(t) for t in lot]))) > 1) | (list(set([type(t) for t in lot]))[0] is not tuple):
            raise Exception('states argument must be a tuple or list of tuples')
    
    
    def find_cities(self, states = None, workers = 1, timeit = True):
        '''
        Get all the cities from a state-level webpage on rmt. Returns a list of tuples of the form: 
            (city page, original state url)
        - states : tuple or list of tuples of the form (state webpage address, state name)
        - workers : int. set to 0 for no threaded processing. 
                         set to -1 to use all available cores 
                         set to >0 to use >0 cores
        - timeit : time the function
        '''
        if timeit: 
            st = time.time() 
        if workers == -1: 
            threads = multiprocessing.cpu_count()
        else: 
            threads = workers 
        if states is None: 
            state_list = self.states 
        elif type(states) is list:
            # make sure every element in the custom list is a tuple
            self.check_for_tups(states)
            state_list = states 
        elif type(states) is tuple:
            state_list = [states]
        else: 
            raise Exception('states argument must be a tuple or list of tuples')
        # if threads = 1 this is the same as running a for loop 
        pool = ThreadPool(threads)
        try: 
            city_links = pool.map(get_cities, [s[0] for s in state_list])
        except: 
            raise Exception('Unable to find the submitted states, if submitting custom state list please make sure spelling is correct. Otherwise check self.states for errors')
        pool = pool.terminate()
        # the pooled process returns a list of list, the loop below extracts them              
        cities = [] 
        for a in city_links:
            for b in a:
                cities = cities + [b]
            #end of for
        # end of for
        if timeit: 
            et = time.time() 
            print('Scraping the states to find cities took %s' % str(et-st))
        return cities 
            


    def find_schools(self, cities, workers = 1, split_thresh = 100, timeit = True):
        '''
        Get all the cities from a state-level webpage on rmt. Returns a list of tuples of the form: 
            (city page, original state url)
        - cities : tuple or list of tuples of the form (citiy webpage address, state url)
        - workers : int. set to 0 for no threaded processing. 
                         set to -1 to use all available cores 
                         set to >0 to use >0 cores
        - split_thresh : int. split the list of cities up into groups of split_thresh to split the workload. 
          does not significantly improve speed but avoids crashes due to overprocessing.
        - timeit : time the function
        '''
        if timeit: 
            st = time.time() 
        if workers == -1: 
            threads = multiprocessing.cpu_count()
        else: 
            threads = workers 
        if type(cities) is list:
            # make sure every element in the custom list is a tuple
            self.check_for_tups(cities)
            city_list = cities 
        elif type(cities) is tuple:
            city_list = [cities]
        else: 
            raise Exception('cities argument must be a tuple or list of tuples')
        # if we have a lot of cities we split the work into chunks 
        if len(city_list) > split_thresh: 
            city_segments = list(chunks(city_list, split_thresh))
        else: 
            city_segments = [city_list]
        # create a list to store the school links 
        school_links = []
        for segment in city_segments: 
            # if threads = 1 this is the same as running a for loop 
            pool = ThreadPool(threads)
            school_links += pool.map(get_schools, segment)
            pool.terminate()
        schools = [] 
        for a in school_links: 
            for b in a: 
                schools += [b]
            # end of for
        # end of for 
        if timeit: 
            et = time.time() 
            print('Scraping the cities to find schools took %s' % str(et-st))
        return schools



    def find_teachers(self, schools, workers = 1, split_thresh = 100, timeit = True):
        '''
        Get all the teachers from a school submission on rmt. Returns a list of tuples of the form: 
            (city page, original state url)
        - cities : tuple or list of tuples of the form (citiy webpage address, state url)
        - workers : int. set to 0 for no threaded processing. 
                         set to -1 to use all available cores 
                         set to >0 to use >0 cores
        - split_thresh : int. split the list of cities up into groups of split_thresh to split the workload. 
          does not significantly improve speed but avoids crashes due to overprocessing.
        - timeit : time the function
        '''
        if timeit: 
            st = time.time() 
        if workers == -1: 
            threads = multiprocessing.cpu_count()
        else: 
            threads = workers 
        if type(schools) is list:
            # make sure every element in the custom list is a tuple
            self.check_for_tups(schools)
            school_list = schools 
        elif type(schools) is tuple:
            school_list = [schools]
        else: 
            raise Exception('cities argument must be a tuple or list of tuples')
        # if we have a lot of cities we split the work into chunks 
        if len(school_list) > split_thresh: 
            school_segments = list(chunks(school_list, split_thresh))
        else: 
            school_segments = [school_list]
        # create a list to store the school links 
        teacher_links = []
        for segment in school_segments: 
            # if threads = 1 this is the same as running a for loop 
            pool = ThreadPool(threads)
            teacher_links += pool.map(get_teachers, segment)
            pool.terminate()
        teachers = [] 
        school_descriptions = pd.DataFrame()
        # each component 'a' is a tuple of the form (teacher list <list>, school description <pandas dataframe>)
        for a in teacher_links:                     
            for b in a[0]: 
                teachers += [b]
            school_descriptions = school_descriptions.append(a[1], ignore_index = True,)                
            # end of for
        # end of for 
        school_descriptions.reset_index()
        if timeit: 
            et = time.time() 
            print('Scraping the schools to find teachers took %s' % str(et-st))
        return teachers, school_descriptions
        


    def extract_reviews(self, teachers, workers = 1, split_thresh = 100, timeit = True):
        '''
        Get all the teachers from a school submission on rmt. Returns a list of tuples of the form: 
            (city page, original state url)
        - cities : tuple or list of tuples of the form (citiy webpage address, state url)
        - workers : int. set to 0 for no threaded processing. 
                         set to -1 to use all available cores 
                         set to >0 to use >0 cores
        - split_thresh : int. split the list of cities up into groups of split_thresh to split the workload. 
          does not significantly improve speed but avoids crashes due to overprocessing.
        - timeit : time the function
        '''
        if timeit: 
            st = time.time() 
        if workers == -1: 
            threads = multiprocessing.cpu_count()
        else: 
            threads = workers 
        if type(teachers) is list:
            # make sure every element in the custom list is a tuple
            self.check_for_tups(teachers)
            teacher_list = teachers 
        elif type(teachers) is tuple:
            teacher_list = [teachers]
        else: 
            raise Exception('cities argument must be a tuple or list of tuples')
        # if we have a lot of cities we split the work into chunks 
        if len(teacher_list) > split_thresh: 
            teacher_segments = list(chunks(teacher_list, split_thresh))
        else: 
            teacher_segments = [teacher_list]
        # create a list to store the school links 
        rating_data = pd.DataFrame()
        for segment in teacher_segments: 
            # if threads = 1 this is the same as running a for loop 
            pool = ThreadPool(threads)
            ratings = pool.map(get_ratings, segment)
            pool.terminate()
            #print(ratings)
            for a in ratings: 
                rating_data = rating_data.append(a, sort = False)
        # end of for 
        if timeit: 
            et = time.time() 
            print('Getting the reviews took %s' % str(et-st))
        return rating_data     
    
    
    def check_last(self, var, full_list): 
        '''
        From full list, get only the values that include and come after the last value in var in the reference database
        '''
        last_val = self.site_map[var].unique()[-1]
        try: # if the value can't be found in the list
            remaining = full_list[[v[0] for v in full_list].index(last_val):]
        except: # return the full list
            remaining = full_list
        return remaining
    
    
    def load_sitemap(self, path = None):
        '''- path : str. the filepath where the data can be found'''
        if path is None: 
            ldir = os.getcwd()
        else: ldir = path 
        if os.path.exists(ldir + '/data/site_map.csv'):                
            self.site_map = pd.read_csv(ldir + '/data/site_map.csv')
        else: 
            print('No existing reference database detected')



    def load_schooldb(self, path = None):
        '''- path : str. the filepath where the data can be found'''
        if path is None: 
            ldir = os.getcwd()
        else: ldir = path 
        if os.path.exists(ldir + '/data/school_database.csv'):
            self.school_database = pd.read_csv(ldir + '/data/school_database.csv')
        else: 
            print('No existing school database detected')


    
    def refresh_database(self, workers = 1, use_existing = None, use_schools = None):
        '''
        Assemble a dataset that contains the full page maps for the home url. Gets all the states, cities, schools, and teachers in a vertical dataset.
        - workers : int. Mutli-threaded option, use 'workers' number of cores to run parallel scraping tasks. 
        - use_existing : pandas DataFrame() object. Existing database to build on top of. 
        - use_schools : pandas DataFrame() object. Existing database with the school description that we can build on top of 
        '''
        if use_existing is None: 
            self.site_map = pd.DataFrame()
            if os.path.exists(os.getcwd() + '/data/site_map.csv'):                
                    use = input("An existing reference database was found, would you like to build on that one? (y/n):")
                    if use == 'y':
                        self.site_map = pd.read_csv(os.getcwd() + '/data/site_map.csv')
                    else: 
                        use_existing = None                        
        else: 
            self.site_map = use_existing
        # if the reference database is not null we pick up at the last observation
        if len(self.site_map) > 0: 
            # keep the states after the last state that was scraped
            states = self.check_last('State', self.states)           
        else: states = self.states
        
        # we can also build on an existing database of school descriptions
        if use_schools is None: 
            self.school_database = pd.DataFrame()
        else:
            self.school_database = use_schools
        
        try:
            # for each state
            for state in states:
                # get all the cities, schools and teachers
                cities = self.find_cities(states = [state], workers = workers)
                # if we're still on the initial state, check which cities might have already been scraped 
                if states.index(state) == 0 and len(self.site_map) > 0: 
                    cities = self.check_last('City', cities)
                # get all the schools in the remaining cities 
                schools = self.find_schools(cities, workers = workers)
                if states.index(state) == 0 and len(self.site_map) > 0: # again, if we're on the first run then this step is needed to eliminate redundant scraping
                    schools = self.check_last('School', schools)
                # get all the teachers in the remaining schools
                teachers, school_descriptions = self.find_teachers(schools, workers = workers)
                if states.index(state) == 0 and len(self.site_map) > 0: 
                    teachers = self.check_last('Teacher', teachers)
                # add them to the list
                self.site_map = self.site_map.append(pd.DataFrame([list(t) for t in teachers], columns = ['Teacher','School','City','State']), sort = False)
                self.school_database = self.school_database.append(school_descriptions, ignore_index = True, sort = False)
                try: 
                    self.site_map = self.site_map.drop('Unnamed: 0', 1)
                except: 
                    pass
                self.site_map = self.site_map.drop_duplicates()
        except Exception as e:
            print(str(e))
            self.site_map.to_csv(os.getcwd() + '/data/site_map.csv')
            self.school_database.to_csv(os.getcwd() + '/data/school_database.csv')
            
        # return the formatted dictionary 
        return self.site_map
    
    
    def get_reviews_by(self, lvl = 'City', for_values = [], prefix = '', workers = 1, overwrite = False): 
        '''
        Get the reviews for all the teachers by any of the nested levels in the reference database
        by = str. 'State' > 'City' (default) > 'School' 
        '''
        # If the site_map hasn't been loaded 
        if len(self.site_map) == 0:
            self.load_reference()
            if len(self.site_map) == 0: 
                raise Exception('No reference database avaialable to guid the scraper, please use the refresh database method or load an existing database')
        
        # if no specific values were selected to scrape
        if len(for_values) == 0:
            for_values = list(self.site_map[lvl].unique)
    
        if lvl == 'Teacher':
            raise Exception('Cannot scrape by "Teacher"')
                
        for val in for_values: 
            # we create layered names that avoid accidental overlap due to repeated school or city names
            if lvl == 'School': 
                index = self.site_map.groupby(lvl)['State','City','School'].first()
                state = index[index['School']==val]['State'].values[0].split('/')[-1]
                city = index[index['School']==val]['City'].values[0].split('/')[-1]
                name = prefix + state + ' ' + city + ' ' + val.split('/')[-1]

            if lvl == 'City': 
                index = self.site_map.groupby(lvl)['State','City'].first()
                state = index[index['City']==val]['State'].values[0].split('/')[-1]
                name =  prefix + state + ' ' + val.splti('/')[-1] 
                
            if lvl == 'State':
                index = self.site_map.groupby(lvl)['State'].first() 
                name =  prefix + val.split('/')[-1]

            # if you have selected not to overwrite data
            if not overwrite: 
                # if the file was already scraped than omit it 
                if os.path.exists(os.getcwd() + '/data/'+name+'.csv'):
                    # then we skip this value
                    continue
            
            # isolate only the teacher links you want
            ds = self.site_map.loc[self.site_map[lvl].isin(for_values)]
            # extract the reviews for those teachers
            reviews = self.extract_reviews(list(zip(ds['Teacher'],ds['School'],ds['City'],ds['State'])), workers = workers)
            # pickle the reviews for the given value
            bn.full_pickle(os.getcwd() + '/data/'+name, reviews)

            

def chunks(l,n):
    '''Break list l up into chunks of size n'''    
    for i in range(0, len(l), n):
        yield l[i:i+n]           
    


def fresh_soup(url):    
    '''
    Collects and parses the page source from a given url, returns the parsed page source 
    - url : the url you wish to scrape
    '''
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.Request(url,headers=hdr) 
    source = urllib.urlopen(req,timeout=10).read() 
    soup = BeautifulSoup(source,"lxml")  
    
    return soup



def get_states(base_url): 
    '''
    Returns the list of states on the ratemyteacher website and their corresponding links. 
    Input must be a front page of ratemyteacher.com
    - base_url : coutnry level url from ratemyteacher.com
    '''
    soup = fresh_soup(base_url)
    # get all the states and turn them into a list of links
    state_links = soup.find_all("div", { "class" : "state"})
    state_pages = [(base_url + '/' + i.text.strip().replace(" ","-"), i.text.strip()) for i in state_links]
    
    return state_pages 



def get_cities(url, verbose = True, base_url = "https://www.ratemyteachers.com/"):
    '''
    Returns the list of the (cities,states) in a given state on the ratemyteacher website and their corresponding links. 
    Input must be the link to the first page of a state page on ratemyteacher. 
    - url : state level url from ratemyteacher.com
    - verbose  : prints the url that was submitted 
    '''
    done = False 
    attempt = 1 
    while done == False:
        #print("Scraping Attempt: " + str(attempt))
        
        try:
            if verbose == True: 
                print(url)

            soup = fresh_soup(url)
                
            city_pages = []
            check = False 
            subattempt = 1 
            # we loop through as many pages as there are. 
            while check == False:     
                # get all the cities and turn them into pages. 
                city_links = soup.find_all("div" , {"class" : "city"})
                city_pages = city_pages + [(i.find(href=True)['href'], url) for i in city_links if i.find(href=True)!= None]
                # many states have more than one page worth of cities, this checks if there is a "next page" link present and, if so, goes to that link.
                next_page = soup.find("a", {"title" : "Next Page"})
                if next_page != None: 
                    print("1.loading " + base_url + next_page['href'])
                    soup = fresh_soup(base_url + next_page['href'])
                else: 
                    check = True 
                if subattempt >= 100:
                    check = True
                subattempt += 1 
        
            # confirm that the process worked
            done = True 
        except: 
            '''
            Since this is a very heavy scraper internet connection insecurity and other issues with the host 
            can increase the likelihood of failure, thus we do not exception out of the code, instead we have
            it keep trying. This results in higher rate of scraped material and keeps the process going. We do this in each function
            '''
            print("Failed to get cities from " + url + " on attempt " + str(attempt) + ", trying again")
            attempt += 1
            soup = fresh_soup(base_url + url)
            if attempt >= 100:
                done = True
        # end of while loop        
    return city_pages



def get_schools(city_state, rating = True, verbose = False, base_url = "https://www.ratemyteachers.com/"): 
    '''
    Returns the list of the schools in a given state on the ratemyteacher website and their corresponding links
    Input must be the link to the first page of a city page on ratemyteacher.com
    - city_state : tuple
    - verbose  : prints the url that was submitted 
    - rating : Boolean; only scrape schools with at least one rating, default value is 'True'
    '''
    url = city_state[0]
    state = city_state[1]    

    done = False 
    attempt = 1 
    while done == False:
        #print("Scraping Attempt: " + str(attempt))

        try:
            if verbose == True: 
                print(url)

            soup = fresh_soup(base_url + url)
          
            school_links = []
            check = False 
            subattempt = 1 
            # we loop through as many pages as there are. 
            while check == False:         
                # get all the schools and turn them into links 
                schools = soup.find_all("div", {"class" : "school"})
                if rating == True:
                    try:
                        school_links = school_links + [(skewl.find("div", {"class" : "info"}).find_all(href=True)[0]['href'],url,state) for skewl in schools if int(re.findall(r'\d+', skewl.find("div", {"class" : "rating_count"}).text)[0]) > 0] 
                    except:
                        print("Skipped " + url)
                elif rating == False: 
                    try:
                        school_links = school_links + [(skewl.find("div", {"class" : "info"}).find_all(href=True)[0]['href'],url,state) for skewl in schools]
                    except:
                        print("Skipped " + url)            
                next_page = soup.find("a", {"title" : "Next Page"})
                if next_page != None: 
                    soup = fresh_soup(base_url + next_page['href'])
                else: 
                    check = True 
                if subattempt >= 100:
                    check = True
                subattempt += 1 
                
            # confirm that the process worked
            done = True 
        except: 
            print("Failed to get schools from " + url + " on attempt " + str(attempt) + ", trying again")
            attempt += 1
            soup = fresh_soup(base_url + url)
            if attempt >= 100:
                done = True
        # end of while loop
    return school_links 



def get_teachers(school_city_state, rating = True, verbose = False, base_url = "https://www.ratemyteachers.com/"): 
    '''
    Returns the list of the teachers in a given school on the ratemyteacher website and their corresponding links
    Input must be the link to the first page of a city page on ratemyteacher.com
    - school_city_state : tuple
    - verbose  : prints the url that was submitted 
    - rating : Boolean; only scrape schools with at least one rating, default value is 'True'
    - get_school_info : Boolean; if True, return a dataset with the info for the school presented on the page
    '''
    url = school_city_state[0]
    city = school_city_state[1]
    state = school_city_state[2]    
    ticker = 1
    done = False 
    attempt = 1 
    while done == False:
        #print("Scraping Attempt: " + str(attempt))

        try:
            if verbose == True: 
                print(url)

            soup = fresh_soup(base_url + url)
            headers = ['School','school_description']
            school_description = [soup.find_all('div', {'class':'description'})[0].text.replace('\n',' ')]
            details = soup.find_all('div',{'class':'detail_item'})
            headers = headers + [f.attrs['class'][1] for f in details]
            school_details = [url] + school_description + [f.text.replace('\n',' ').strip() for f in details]
            school_details = pd.DataFrame(school_details, headers).transpose()              

            teacher_links = []
            check = False 
            subattempt = 1 
            # we loop through as many pages as there are. 
            while check == False:  
                #print("Sub attempt " + str(subattempt))
                # we're recording the hrefs before moving to the next page. 
                teachers = soup.find_all("div", {"class" : "teacher"})
                if rating == True:
                    teacher_links = teacher_links + [(t.find_all(href = True)[0]['href'],url,city,state) for t in teachers if int(re.findall(r'\d+', t.find("div", {"class":"rating_count"}).text)[0]) > 0]
                elif rating == False: 
                    teacher_links = teacher_links + [(t.find_all(href = True)[0]['href'],url,city,state) for t in teachers]    
                
                next_page = soup.find("a", {"title" : "Next Page"})
                if next_page != None: 
                    soup = fresh_soup(base_url + next_page['href'])
                    #print("There is another page, " + str(ticker))
                    ticker += 1 
                else: 
                    check = True 
                if subattempt >= 10:
                    check = True
                subattempt += 1 

            # confirm that the process worked
            done = True 
        except: 
            print("Failed to get teachers from " + url + " on attempt " + str(attempt) + ", trying again")
            attempt += 1
            soup = fresh_soup(base_url + url)
            if attempt >= 100:
                done = True            
        # end of while loop            
    return teacher_links, school_details



def get_ratings(teacher_school_city_state, verbose = False, base_url = "https://www.ratemyteachers.com/"):
    '''
    Returns a pandas.DataFrame() object with the ratings for a teacher in a course. 
    Input must be the link to the first page of a teacher's page on ratemyteacher.com
    - teacher_school_city_state : tuple
    - verbose  : prints the url that was submitted 
    - rating : Boolean; only scrape schools with at least one rating, default value is 'True'
    ''' 
    url = teacher_school_city_state[0]
    school = teacher_school_city_state[1]   
    city = teacher_school_city_state[2]
    state = teacher_school_city_state[3]

    ticker = 1 
    done = False 
    attempt = 1 
    while done == False:
        #print("Scraping Attempt: " + str(attempt))

        try:
            if verbose == True:
                print(url)

            soup = fresh_soup(base_url + url)
            
            l = 0    
            df = pd.DataFrame(columns = ['Rating','Text', 'State', 'City', 'School','Teacher','TeachDes','DateTime','SubmittedBy'])
             
            teacher_description = soup.find("div", {"class" : "description"})
            try:
                teacher_description = teacher_description.text.strip().replace("\n"," ")
            except:
                teacher_description = None                    
            
            check = False                                 
            subattempt = 1 
            while check == False: 
                # we find all the reviews 
                reviews = soup.find_all("div", {"itemprop" : "review"})
                # here we are going to record all the review data before moving on to the next page. 
                for r in reviews:
                    # we find all the reviews
                    attributes = [(i.text.strip().split("\n")[0],i.text.strip().split("\n")[len(i.text.strip().split("\n"))-1]) for i in r.find_all("div", {"class" : "attribute"})]
                    # create an empty dictionary because the attributes are easier to assign. 
                    row = {}
                    # place the attributes in a dictionary
                    for i in attributes:
                        if i[1] not in df.columns:
                            df[i[1]] = None 
                        row[i[1]]=i[0]
                    # automatically maps the labels with the values in the dataset 
                    df = df.append(row, ignore_index = True) 
                    try:
                        df.loc[l]['Rating'] = re.findall(r'\d+', str(r.find("div", {"class":"rateit-selected"})))[1]
                    except:
                        pass
                    try:
                        df.loc[l]['Text'] = r.find("span", {"class":"text"}).text.strip()
                    except:
                        pass                            
                    try:
                        df.loc[l]['SubmittedBy'] = r.find('div', {"class" : "submitted_by"}).text.strip()
                    except:
                        pass             
                    
                    df.loc[l]['DateTime'] = r.find("span", {"class" : "date"})['datetime'] 
                    df.loc[l]['TeachDes'] = teacher_description
                    df.loc[l]['State'] = state
                    df.loc[l]['City'] = city
                    df.loc[l]['School'] = school
                    df.loc[l]['Teacher'] = url                 
                    l+=1 
                    # end of reviews loop           
                next_page = soup.find("a", {"title" : "Next Page"})
                if next_page != None: 
                    soup = fresh_soup(base_url + next_page['href'])
                    #print("There is another page, " + str(ticker))
                    ticker += 1
                else: 
                    check = True 
                # end of while
                if subattempt >= 10:
                    check = True
                subattempt += 1 
                
            # confirm that the process worked
            done = True        
            
        except: 
            print("Failed to get ratings from " + url + " on attempt " + str(attempt) + ", trying again")
            attempt += 1
            soup = fresh_soup(base_url + url)
            if attempt >= 100:
                done = True            
        # end of while loop            
    return df