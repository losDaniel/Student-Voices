import matplotlib, random
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display_html
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pyLDAvis
import pyLDAvis.gensim
import re 

def display_side_by_side(*args):
    '''Display the given tables side by side'''
    html_str=''
    for df in args:
    	if type(df) == list:
    		for t in df: 
    			html_str+=t.to_html()
    	else:
	        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


def get_random_color_list(length, name=True): 
    hex_colors_dic = {}
    rgb_colors_dic = {}
    hex_colors_only = []
    for name, hex in matplotlib.colors.cnames.items():
        hex_colors_only.append(hex)
        hex_colors_dic[name] = hex
        rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)

    if name: 
        return random.sample(list(rgb_colors_dic.keys()), length)
    if not name:
        return random.sample(hex_colors_only, length)
        

def plot_cumulative_rating_dist(site_data, reverse_axis=True):
    '''Plot the cumulative distributions of Review Ratings highlighting bins in rectangles'''
    
    vc = site_data['Rating'].value_counts()
    cum_ratings_dist = pd.DataFrame(vc.sort_index(ascending=False) / vc.sum())['Rating'].cumsum()
    cum_ratings_dist = cum_ratings_dist.reset_index().rename(columns={'Rating':'% of Reviews', 'index':'Rating'})

    fig = go.Figure()
    fig = fig.add_trace(go.Scatter(x=cum_ratings_dist["Rating"], y=cum_ratings_dist["% of Reviews"], name='% of reviews',
                  line = dict(color='black', width=2)))

    bins = list(site_data['Bins'].unique())

    #color_list = get_random_color_list(len(bins))
    color_list = ['cornflowerblue', 'hotpink', 'gold', 'orange', 'crimson', 'grey', 'violet']

    bin_num = 0 
    for val in sorted(bins): 
        fig.add_shape(
                # unfilled Rectangle
                    type="rect",
                    x0=site_data.loc[site_data['Bins']==val,'Rating'].min(),
                    y0=0,
                    x1=site_data.loc[site_data['Bins']==val,'Rating'].max(),
                    y1=1.1,
                    line=dict(
                        color=color_list[bin_num],
                        dash="dashdot",
                    ),
                    name=str(val),
                    #showlegend=True,
                )

        col = site_data.loc[site_data['Bins']==val,'Rating']
        fig.add_trace(go.Bar(        
            # Add invisible bars with the same colors as the shapes so the squares can be added to the legend 
            width = [0],
            x = [col.median()],
            y = [len(col)/len(site_data)],
            text = '{:.2f}'.format(100*len(col)/len(site_data))+'%',
            name = str(val)+': ''{:.2f}'.format(100*len(col)/len(site_data))+'%',
            marker_color=color_list[bin_num],
            visible = True,
            showlegend=True,
        ))
        
        bin_num+=1
        
    fig.update_layout(title='Cumulative Distribution by Rating',
                      xaxis_title='Ratings in reverse order',
                      yaxis_title='% of Reviews',
                      showlegend=True,
                      )
    if reverse_axis:
        fig.update_layout(xaxis = dict(autorange = "reversed"))

    return fig


def plot_rating_dist(site_data, reverse_axis=False):
    '''Plot the cumulative distributions of Review Ratings highlighting bins in rectangles'''
    
    vc = site_data['Bins'].value_counts()
    ratings_dist = pd.DataFrame(vc.sort_index(ascending=False) / vc.sum())
    ratings_dist = ratings_dist.reset_index().rename(columns={'Rating':'% of Reviews', 'index':'Rating'})
    ratings_dist.columns = ['Bin','% of Ratings']

    fig = go.Figure()
    fig.add_trace(go.Bar(        
        x = [str(r) for r in ratings_dist['Bin']],
        y = ratings_dist['% of Ratings'],
        text = ['{:.2f}'.format(f*100)+'%' for f in ratings_dist['% of Ratings']],
        visible = True,
        marker_color='black',
    ))
        
    fig.update_layout(title='% Reviews by Rating Bin',
                      xaxis_title='Rating Bin',
                      yaxis_title='% of Reviews',
                      showlegend=False,
                      )
    if reverse_axis:
        fig.update_layout(xaxis = dict(autorange = "reversed"))

    return fig


def chart_review_lengths(tables, save=None):
    name = [] 
    counts = [] 
    p25 = []
    p50 = [] 
    p75 = [] 
    for t in tables: 
        name.append(t.columns[0])
        counts.append(int(t.loc['count'].values[0].replace(',','')))
        p25.append(int(t.loc['25%'].values[0].replace(',','')))
        p50.append(int(t.loc['50%'].values[0].replace(',','')))
        p75.append(int(t.loc['75%'].values[0].replace(',','')))
    graph_data = pd.DataFrame({'name':name,'count':counts,'p25':p25,'p50':p50,'p75':p75}).sort_values('count')

    plt.rcParams['figure.figsize'] = 7, 7
    plt.rcParams['axes.facecolor']='lightblue'
    plt.rcParams['figure.facecolor']='white'
    plt.plot(graph_data['count'],graph_data['p25'], linestyle='--', marker='x')
    plt.plot(graph_data['count'],graph_data['p50'], linestyle='--', marker='x')
    plt.plot(graph_data['count'],graph_data['p75'], linestyle='--', marker='x')
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.yticks(np.arange(0, max(graph_data['p75'])+10, 10))
    plt.ylabel('Review Length')
    plt.xlabel('Number of Observations')
    plt.legend()
    if save is not None: 
        plt.savefig(save)
    plt.show()   
    
    

import seaborn as sns
    
def plot_restricted_review_dists(data, save=None):
    plt.rcParams['axes.facecolor']= 'white'
    plt.rcParams['figure.facecolor']= 'white'
    plt.rcParams['font.size'] = 18
    # create a column containing the number of reviews a given teacher recieved
    data['num_reviews'] = data.groupby('Teacher')['FID'].transform('count')
    
    # we will also bin these differently for the histograms
    bins = list(range(0,101,10))
    
    min_num_reviews = [1,3,5,10,25,50]
       
    fig, axes = plt.subplots(2,3)
    plt.subplots_adjust(wspace=0.25, hspace=0.25, left=0.1, bottom=0.22, right=0.96, top=0.92)
    axes = axes.flatten()
    fig.set_size_inches(20, 15)
    #fig.suptitle('Distribution of Features')
    
    
    for i,n in enumerate(min_num_reviews):        
        lbl = 'Min Reviews %d, Total Obs are %d' % (n, (len(data.loc[data['num_reviews']>n])))
        graph_data = pd.cut(data.loc[data['num_reviews']>n, 'Rating'], bins, labels=[int(i) for i in bins[1:]]).dropna()
        vc = pd.DataFrame(graph_data.value_counts().reset_index())
        vc.columns = [lbl, '']
        sns.lineplot(vc[lbl], vc[''], ax=axes[i])
        if i == 0 or i == 3: 
             axes[i].set_ylabel('# Reviews')
    #    sns.distplot(pd.cut(data.loc[data['num_reviews']>n, 'Rating'], bins, labels=[int(i) for i in bins[1:]]).dropna(), axlabel=lbl, ax=axes[i])
    #    sns.distplot(data.loc[data['num_reviews']>n,'Rating'], bins = bins, axlabel=lbl, ax=axes[i])
    #    axes[i].axvline(data.loc[data['num_reviews']>n,'Rating'].mean(),linewidth=1)
    #    axes[i].axvline(data.loc[data['num_reviews']>n,'Rating'].median(),linewidth=1, color='r')
    
    if save is not None: 
        fig.savefig(save)
        
        
# Function to describe the topics 
def get_topic_reviews(topic, reviews, poolsize =50, samplesize = 20, fsize = (8,8), boundary=False, graph=True, get_series=None, save_at=None, print_topics=False):
    # set graph colors 
    plt.rcParams['axes.facecolor']= 'gold'
    plt.rcParams['figure.facecolor']= 'lightgray'

    # get the reviews for the specified topic 
    topic_reviews = reviews.loc[reviews['Dominant_Topic']==topic].sort_values(by=['Dominant_Topic','Topic_Perc_Contrib'], ascending=[False, False])
    
    print('There are '+str(len(topic_reviews))+' reviews dominated by this Topic')
    # get the keywords for the topics
    keywords = ', '.join(list(topic_reviews[:1]['Keywords'].values))

    # show the key words 
    print(keywords)

    if graph is True:
        # instantiate the plot 
        fig = plt.figure(1, figsize = fsize)
        ax = fig.add_subplot(111)
        
        # plot the distribution of the percentage contribution of the topic to each review that has the given topic as a dominant topic
        sns.distplot(topic_reviews['Topic_Perc_Contrib'], ax=ax)
        plt.show()
    
    if not boundary: 
        # select a random sample of reviews from a pool of size poolsize  
        sample = random.sample(list(topic_reviews['Text'][:poolsize].values),samplesize)
    elif boundary: 
        
        if get_series is not None: 
            full_sample = []
            for ps in get_series: 
                for review in list(topic_reviews['Text'][:ps].values)[-samplesize:]:
                    full_sample.append(review)
            
            indexes = [i for i in [list(range(samp-2, samp+1)) for samp in get_series]]
            index_list = []
            for lidx in indexes:
                for i in lidx:
                    index_list.append(i)
            
            sample = pd.DataFrame({'Text':full_sample}, index=index_list)

            if save_at is not None:
                sample.to_csv(save_at)

        else: 
            sample = list(topic_reviews['Text'][:poolsize].values)[-samplesize:]
        
    if print_topics:
        # display each review
        print('\n')
        for s in sample:
            print(s.replace('Submitted by a student','').replace('Submitted by a parent','').strip(),'\n')
    
    return topic_reviews


def plot_dominant_topics(reviews, ex=None, tc=None, plt_fontsize=14, leg_fontsize=14, newlinechars=120):
    # set graph colors 
    plt.rcParams['axes.facecolor']= 'gold'
    plt.rcParams['figure.facecolor']= 'lightgray'
    plt.rcParams['font.size']= plt_fontsize    
    plt.rcParams["legend.fontsize"] = leg_fontsize
    
    # create the elements of the lengend with the titles 
    patchList = []
    i = 0 
    for key in ex: # for each topic 
        # we make the legend with the detailed descriptions of each topic and insert a new line every `newlinechars`# of characters
        data_key = mpatches.Patch(color=sns.color_palette('husl',len(ex))[i], label=str(i)+': '+re.sub("(.{"+str(newlinechars)+"})", "\\1\n", list(ex)[i], 0, re.DOTALL))
        patchList.append(data_key)
        i+=1

    # get the value counts for the dominant topics to graph 
    vc = reviews['Dominant_Topic'].value_counts().sort_index()
    fig = plt.figure(1, figsize=(5,5))
    
    fig.suptitle('Number of Reviews Per Topic\n(** next to description = topics that may not be consistent)')
    
    ax = fig.add_subplot(111)
    # create the barplot with the dominant topic distributions
    sns.barplot(list(vc.keys().astype(int)), list(vc.values), ax=ax, palette=sns.color_palette('husl',len(ex)))

    if tc is not None: 
        ax.set_ylabel('(bars) number of observations')
        ax2 = ax.twinx()
        ax2.set_ylabel('(line) coherence score')
        # graph line plot with coherence scores
        sns.lineplot(list(vc.keys().astype(int)), tc, ax=ax2)
    
    # place the legend below the graph 
    plt.legend(handles=patchList, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True)
    plt.show()
    
    
def save_topic_visualization(model, docs, dic, path):
    '''
    Use the pyLDAvis module to generate a graph of the distributed topics. 
    - model : lda model to use (lda or ldamulticore)
    - docs : we want to visualize 
    - dic : the dictionary from the model 
    - path : str. the path and filename for the graph, include ".html" extension in the name. Does not save when set to None 
    - show : boolean. Set to true to display the HTML version of the graph 
    '''
    text = [dic.doc2bow(doc) for doc in docs]
    lda_vis = pyLDAvis.gensim.prepare(model, text, dic, sort_topics = True)
    # save it to an html format    
    try: 
        pyLDAvis.save_html(lda_vis, path)
    except Exception as e: 
        print(str(e))
        print('Make sure you are entering a valid path with the .html extension')