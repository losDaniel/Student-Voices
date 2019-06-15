# -*- coding: utf-8 -*-
"""
@author: Carlos Valcarcel

Methods to help process and visualize analysis results 
"""
from IPython.display import display_html
from IPython.core.display import display
import pandas as pd
import os
import pyLDAvis
import pyLDAvis.gensim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
from sklearn.decomposition import PCA
from seaborn import color_palette,palplot
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import random 
from pyvis.network import Network
import re 



# Function to describe the topics 
def get_topic_reviews(topic, reviews, poolsize =50, samplesize = 20, fsize = (8,8)):
    # set graph colors 
    plt.rcParams['axes.facecolor']= 'gold'
    plt.rcParams['figure.facecolor']= 'lightgray'

    # get the reviews for the specified topic 
    topic_reviews = reviews.loc[reviews['Dominant_Topic']==topic].sort_values(by=['Dominant_Topic','Topic_Perc_Contrib'], ascending=[False, False])
    
    # get the keywords for the topics
    keywords = ', '.join(list(topic_reviews[:1]['Keywords'].values))

    # show the key words 
    print(keywords)

    # instantiate the plot 
    fig = plt.figure(1, figsize = fsize)
    ax = fig.add_subplot(111)
    
    # plot the distribution of the percentage contribution of the topic to each review that has the given topic as a dominant topic
    sns.distplot(topic_reviews['Topic_Perc_Contrib'], ax=ax)
    plt.show()
    
    # select a random sample of reviews from a pool of size poolsize  
    sample = random.sample(list(topic_reviews['Text'][:poolsize].values),samplesize)
    # display each review
    print('\n')
    for s in sample:
        print(s.replace('Submitted by a student','').strip(),'\n')
    
    return topic_reviews



def plot_dominant_topics(reviews, ex, tc=None, plt_fontsize=14, leg_fontsize=14, newlinechars=120):
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
        data_key = mpatches.Patch(color=sns.color_palette('husl',len(ex))[i], label=str(i)+': '+re.sub("(.{"+str(newlinechars)+"})", "\\1\n", list(ex.values())[i], 0, re.DOTALL))
        patchList.append(data_key)
        i+=1

    # get the value counts for the dominant topics to graph 
    vc = reviews['Dominant_Topic'].value_counts().sort_index()
    fig = plt.figure(1, figsize=(15,15))
    
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
    
    # place the legen below the graph 
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
    lda_vis = pyLDAvis.gensim.prepare(model, text, dic, sort_topics = False)
    # save it to an html format    
    try: 
        pyLDAvis.save_html(lda_vis, path)
    except Exception as e: 
        print(str(e))
        print('Make sure you are entering a valid path with the .html extension')


        
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

    

def split_tables(syns,word_list,sims=None):
    '''Split a list of words into side by side tables'''
    if sims is None: 
        syns_2 = {} 
        for x in syns: 
            if x in syns_2: 
                syns_2[x] += 1 
            else: 
                syns_2[x] = 1 
        
        delt = len(syns)
        syns = list(set(syns))
        delt -= len(syns)
        ts = int(len(syns)/(abs(len(word_list)-delt)))
        syns = [x for x in syns_2.keys()]
        sims = [x for x in syns_2.values()]
    else:
        ts = int(len(syns)/len(word_list))
    
    tbls = []
    for i in range(0,len(word_list)):
#        if sims is None:
 #           tbls.append(pd.DataFrame({'Word':syns[ts*i:ts*(i+1)]}, index = range(ts*i,ts*(i+1))))        
  #      else: 
            tbls.append(pd.DataFrame({'Word':syns[ts*i:ts*(i+1)],'Score':sims[ts*i:ts*(i+1)]}, index = range(ts*i,ts*(i+1))))
    return tbls



def graph_dist(data, cols, rownum = 2, mean_line = True, median_line = True):
    features = cols
    plot_num = int(int(rownum * np.ceil(float(len(cols))/rownum))/rownum)
    fig, axes = plt.subplots(rownum, plot_num)
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.22, right=0.96, top=0.92)
    axes = axes.flatten()
    fig.set_size_inches(18, 10)
    fig.suptitle('Distribution of Features')

    for i, feature in enumerate(features):       
        lbl = feature + ' (' + str(len(data[feature].dropna())) + ')'
        graph_data = data.rename(columns={feature:lbl})[lbl].dropna().copy()
        sns.distplot(graph_data, label= feature, ax=axes[i]).set(xlim=(graph_data.min(), graph_data.max()),)
        if mean_line: 
            axes[i].axvline(graph_data.mean(),linewidth=1)
        if median_line: 
            axes[i].axvline(graph_data.median(),linewidth=1, color='r')


            
def horizontal_barchart(x, y, title='', save=None, dims =(15,15), afc='white', ffc='white', rot=65, fs=12):
    
    plt.rcParams.update({'font.size': 16})
    # set the background color of the plot to white 
    plt.rcParams['axes.facecolor']= afc
    plt.rcParams['figure.facecolor']= ffc

    fig, ax = plt.subplots(figsize = dims)

    ax.barh(x, y, align='center', ecolor='black', color = 'lightcoral', height = 0.5)
    
    # values next to bars 
    for i, v in enumerate(y):
        ax.text(v + 3, i + .25, "{:,.0f}".format(v), color='blue', fontsize = fs, fontweight='bold')

    ax.set_yticks(x)
    ax.set_yticklabels(x, fontsize = fs)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_title(title)
    plt.xlabel('number of reviews')
    plt.yticks(rotation=rot)

    
    if save is not None: 
        fig.savefig(save) 
        plt.close(fig)
                      

            
def chart_review_lengths(tables):
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
    plt.savefig(os.getcwd()+'/graphs/review length percentile distributions.png')
    plt.show()            
            
            

def bar_dists(dataframe, num_plots, save=None):
    # set the background color of the plot to white 
    plt.rcParams['axes.facecolor']='lightblue'
    plt.rcParams['figure.facecolor']='white'

    # create a figure with any number of plots 
    fig, axes = plt.subplots(num_plots,1)

    # adjust the spacing of the plots 
    plt.subplots_adjust(wspace=1, hspace=0.2, left=0.1, bottom=0.22, right=0.96, top=1)

    axes = axes.flatten()
    fig.set_size_inches(18, 10)
    fig.suptitle('Distribution of Topics')
    # set the limit for the y-axis
    ylim = dataframe['occurrence'].max()

    for i, ds in enumerate(np.array_split(dataframe.sort_values('occurrence'),num_plots)): 
        # make a barplot of the subdataset 
        gr = sns.barplot(ds['seeds'],ds['occurrence'], ax=axes[i])
        position = 0 
        gr.set(ylim=(0, ylim))
        for idx, row in ds.iterrows(): 
            gr.text(position,row['occurrence']*.9, row['occurrence'],color='black',fontsize=14,ha='center')
            position += 1
        
    if save is not None: 
        fig.savefig(save)

        

def pca_results(good_data, pca, width = 14, height = 8):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (width,height))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)



# Just a clunky little function to generate a legend (given color pallette and labels) as
# an independent matplotlib figure
def keyplot(pal, names):
    n = len(pal)
    rows = n / 5
    if n%5 > 0:
        rows += 1
    f, axes = plt.subplots(int(rows), 1, figsize=(15, int(.5*rows)))
    if rows ==1:
        axes = [axes,]
    for idx,ax in enumerate(axes):
        current_pal = pal[idx*5:(idx*5)+5]
        if len(current_pal)<5:
            
            current_pal += ['white']*(5-len(current_pal))
        current_names = names[idx*5:(idx*5)+5]
        ax.imshow(np.arange(5).reshape(1, 5),
                  cmap=mpl.colors.ListedColormap(list(current_pal)),
                  interpolation="nearest", aspect="auto")
        ax.set_xticks(np.arange(5) - .5)
        ax.set_yticks([-.5, .5])
        for i,name in enumerate(current_names):
            ax.annotate(name,(-.45+i,0.1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        

# method to get 2d projection of the entire dictionary 
def project_dictionary(model, model_dict):
    '''Project a model dictionary onto a 2d space and return a dataframe with the vectors and words'''
    # get the vectors for every word in the dictionary 
    vectorized_dict = [model.wv.get_vector(model_dict[w]) for w in model_dict]

    # get the two dimensional projection of all the word vectors 
    pca = PCA(n_components = 2)
    pca_data = pd.DataFrame(pca.fit_transform(vectorized_dict), columns = ['X','Y'])
    pca_data['Words'] = [model_dict[w] for w in model_dict]

    return pca_data  



# method to get the similarity sub-sample for a given word
def add_degrees_of_separation(word, model, dos = 2): 
    '''Submit a word and model to get the words within dos degrees of separation from the given word'''    
    degrees = {} 
    degrees[word] = [0]
    for s in model.wv.similar_by_word(word):
        degrees[s[0]] = 1
    for i in range(1, dos):
        checklist = list(degrees.keys())
        for w in checklist:
            if degrees[w] == i: 
                for nw in [s[0] for s in model.wv.similar_by_word(w)]:
                    if nw not in degrees: 
                        degrees[nw] = i+1 
    return degrees


# define some categorical fonts 
main_word_font_a = {'family': 'serif',
                  'color':  'black',
                  'weight': 'normal',
                  'size': 25}

sim_word_font_1_a = {'family': 'serif',
                 'color':  'chocolate',
                 'weight': 'normal',
                 'size': 15}

sim_word_font_2_a = {'family': 'serif',
                 'color':  'goldenrod',
                 'weight': 'normal',
                 'size': 9}

main_word_font_a = {'family': 'serif',
                  'color':  'black',
                  'weight': 'normal',
                  'size': 15}

sim_word_font_1_b = {'family': 'serif',
                 'color':  'green',
                 'weight': 'normal',
                 'size': 11}

sim_word_font_2_b = {'family': 'serif',
                 'color':  'seagreen',
                 'weight': 'normal',
                 'size': 9}

# method to label points with words and apply font by type
def label_point(x, y, val, grp, ax, mwf = main_word_font_a, swf = sim_word_font_1_b, swf2 = sim_word_font_2_b):
    a = pd.concat({'x': x, 'y': y, 'val': val, 'grp': grp}, axis=1)
    for i, point in a.iterrows():
        if point['grp'] == 0:
            ax.text(point['x']+.02, point['y'], str(point['val']), fontdict = mwf)
        elif point['grp'] == 1: 
            ax.text(point['x']+.02, point['y'], str(point['val']), fontdict = swf)
        elif point['grp'] == 2: 
            ax.text(point['x']+.02, point['y'], str(point['val']), fontdict = swf2)

            
# method to plot the words  
def plot_word_similarities(word_vectors, title = 'Similar Words', save=None): 
    '''
    Create xy scatter plot of words in word_vectors
    - word_vectors : pd.DataFrame with 'X','Y', 'Words' columns
    - title : graph title
    '''

    # set the background color of the plot to white 
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['figure.facecolor']='white'

    ax = sns.lmplot('X', # Horizontal axis
                    'Y', # Vertical axis
                    data=word_vectors, # Data source
                    fit_reg=False, # Don't fix a regression line
                    size = 10,
                    aspect =2) # size and dimension

    plt.title(title)
    # Set x-axis label
    plt.xlabel('X')
    # Set y-axis label
    plt.ylabel('Y')

    label_point(word_vectors.X, word_vectors.Y, word_vectors.Words, word_vectors.Degrees, plt.gca())
    if save is not None: 
        plt.savefig(save)
    
    
def setup_network_graph(topic_of_the_day, model, seeds):
    db = pd.DataFrame()
    # get the synonyms and weights for the key words in the topics 
    for w in seeds[topic_of_the_day]: 
        syns = model.wv.similar_by_word(w)
        targets = [] 
        weights = [] 
        for s in syns: 
            targets.append(s[0])
            weights.append(s[1])
        ds = pd.DataFrame({'Targets':targets,'Weights':weights})
        ds['Source'] = w
        db = db.append(ds, sort=True)

    # for all other words, find how they relate to one another, but don't include any new words 
    existing_targets = list(db['Targets'].unique())
    existing_sources= list(db['Source'].unique())
    # for all the words that aren't key words 
    for w in [v for v in list(db['Targets'].unique()) if v not in seeds[topic_of_the_day]]:
        syns = model.wv.similar_by_word(w)
        targets = [] 
        weights = [] 
        for s in syns: # find and go through the synonyms  
            # only consider synonyms that are already within the list of considered words 
            if s[0] in existing_targets and s[0] in existing_sources: 
                targets.append(s[0])
                weights.append(s[1])
            ds = pd.DataFrame({'Targets':targets,'Weights':weights})
            ds['Source'] = w
            db = db.append(ds, sort=True)

    # create a column identifying unique pairs 
    db['pair'] = db.apply(lambda row: ','.join(list(set([row['Source'],row['Targets']]))),1)
    # then group them, adding their weights together
    db = pd.merge(db.groupby('pair', as_index = False)[['Source','Targets']].first(), # also use the first ordering of the words appearances 
             db.groupby('pair', as_index = False)[['Weights']].sum(),
             on = 'pair',
             how = 'inner')
    # drop the ordering variable 
    db = db.drop('pair', 1)
    
    return db    


def draw_network_graph(db, save):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # set the physics layout of the network
    net.barnes_hut()

    sources = db['Source']
    targets = db['Targets']
    weights = db['Weights']

    edge_data = zip(sources, targets, weights)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]

        net.add_node(src, src, title=src)
        net.add_node(dst, dst, title=dst)
        net.add_edge(src, dst, value=w)

    neighbor_map = net.get_adj_list()

    # add neighbor data to node hover data
    for node in net.nodes:
        node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])

    net.save_graph(save)
    


# PLOTTING FOR JUPYTER SPECFICALLY 

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

def num_topics_by_theme(topic_dataset,theme_dict):
    # get the number of topics for each row 
    topic_dataset['num_topics'] = topic_dataset.count(axis=1)

    for k in theme_dict:
        if '4' in theme_dict[k]: theme_dict[k] = '4'

    topic_dataset['theme'] = topic_dataset[0].map(theme_dict)

    distribution = topic_dataset.groupby(['theme','num_topics'], as_index=False).count()[['theme','num_topics',0]]
    distribution = distribution.pivot(index = 'num_topics', columns ='theme',values =0)
    for c in distribution.columns:
        distribution[c] = distribution[c]/distribution[c].sum()
    distribution.iplot(kind='bar', xTitle='#Topics',yTitle='count', title='Number of Topics per Review by Theme')
    

    
def plot_parent_reviews(data, range_indices):
    list1 = list(data.loc[data['SubmittedBy']=='Submitted by a Parent'].index)

    # get the indices of reviews that made it into the present corpus 
    range_indices = bn.loosen(os.getcwd() + '/data/by_rating_range.pickle')
    indices = data.loc[range_indices['[0, 35)'],'Review_Length']
    filtered_index = indices[indices>100].index
    result = np.intersect1d(list1, filtered_index)

    # get the parent reviews 
    parent_reviews = [i for i, e in enumerate(filtered_index) if e in result]

    plotdf = topic_dataset.ix[parent_reviews].rename(columns={0:'DomTop'})
    plotdf['DomTop'].iplot(kind='hist', xTitle='topic',yTitle='count', title='Where did the Parent Reviews Go?')
    
    
# Data Shader Color Pallette 
# Lookup of web color names to their hex codes.
color_lookup = {'aliceblue': '#F0F8FF', 'antiquewhite': '#FAEBD7',
                'aqua': '#00FFFF', 'aquamarine': '#7FFFD4',
                'azure': '#F0FFFF', 'beige': '#F5F5DC',
                'bisque': '#FFE4C4', 'black': '#000000',
                'blanchedalmond': '#FFEBCD', 'blue': '#0000FF',
                'blueviolet': '#8A2BE2', 'brown': '#A52A2A',
                'burlywood': '#DEB887', 'cadetblue': '#5F9EA0',
                'chartreuse': '#7FFF00', 'chocolate': '#D2691E',
                'coral': '#FF7F50', 'cornflowerblue': '#6495ED',
                'cornsilk': '#FFF8DC', 'crimson': '#DC143C',
                'cyan': '#00FFFF', 'darkblue': '#00008B',
                'darkcyan': '#008B8B', 'darkgoldenrod': '#B8860B',
                'darkgray': '#A9A9A9', 'darkgreen': '#006400',
                'darkgrey': '#A9A9A9', 'darkkhaki': '#BDB76B',
                'darkmagenta': '#8B008B', 'darkolivegreen': '#556B2F',
                'darkorange': '#FF8C00', 'darkorchid': '#9932CC',
                'darkred': '#8B0000', 'darksage': '#598556',
                'darksalmon': '#E9967A', 'darkseagreen': '#8FBC8F',
                'darkslateblue': '#483D8B', 'darkslategray': '#2F4F4F',
                'darkslategrey': '#2F4F4F', 'darkturquoise': '#00CED1',
                'darkviolet': '#9400D3', 'deeppink': '#FF1493',
                'deepskyblue': '#00BFFF', 'dimgray': '#696969',
                'dimgrey': '#696969', 'dodgerblue': '#1E90FF',
                'firebrick': '#B22222', 'floralwhite': '#FFFAF0',
                'forestgreen': '#228B22', 'fuchsia': '#FF00FF',
                'gainsboro': '#DCDCDC', 'ghostwhite': '#F8F8FF',
                'gold': '#FFD700', 'goldenrod': '#DAA520',
                'gray': '#808080', 'green': '#008000',
                'greenyellow': '#ADFF2F', 'grey': '#808080',
                'honeydew': '#F0FFF0', 'hotpink': '#FF69B4',
                'indianred': '#CD5C5C', 'indigo': '#4B0082',
                'ivory': '#FFFFF0', 'khaki': '#F0E68C',
                'lavender': '#E6E6FA', 'lavenderblush': '#FFF0F5',
                'lawngreen': '#7CFC00', 'lemonchiffon': '#FFFACD',
                'lightblue': '#ADD8E6', 'lightcoral': '#F08080',
                'lightcyan': '#E0FFFF', 'lightgoldenrodyellow': '#FAFAD2',
                'lightgray': '#D3D3D3', 'lightgreen': '#90EE90',
                'lightgrey': '#D3D3D3', 'lightpink': '#FFB6C1',
                'lightsage': '#BCECAC', 'lightsalmon': '#FFA07A',
                'lightseagreen': '#20B2AA', 'lightskyblue': '#87CEFA',
                'lightslategray': '#778899', 'lightslategrey': '#778899',
                'lightsteelblue': '#B0C4DE', 'lightyellow': '#FFFFE0',
                'lime': '#00FF00', 'limegreen': '#32CD32',
                'linen': '#FAF0E6', 'magenta': '#FF00FF',
                'maroon': '#800000', 'mediumaquamarine': '#66CDAA',
                'mediumblue': '#0000CD', 'mediumorchid': '#BA55D3',
                'mediumpurple': '#9370DB', 'mediumseagreen': '#3CB371',
                'mediumslateblue': '#7B68EE', 'mediumspringgreen': '#00FA9A',
                'mediumturquoise': '#48D1CC', 'mediumvioletred': '#C71585',
                'midnightblue': '#191970', 'mintcream': '#F5FFFA',
                'mistyrose': '#FFE4E1', 'moccasin': '#FFE4B5',
                'navajowhite': '#FFDEAD', 'navy': '#000080',
                'oldlace': '#FDF5E6', 'olive': '#808000',
                'olivedrab': '#6B8E23', 'orange': '#FFA500',
                'orangered': '#FF4500', 'orchid': '#DA70D6',
                'palegoldenrod': '#EEE8AA', 'palegreen': '#98FB98',
                'paleturquoise': '#AFEEEE', 'palevioletred': '#DB7093',
                'papayawhip': '#FFEFD5', 'peachpuff': '#FFDAB9',
                'peru': '#CD853F', 'pink': '#FFC0CB',
                'plum': '#DDA0DD', 'powderblue': '#B0E0E6',
                'purple': '#800080', 'red': '#FF0000',
                'rosybrown': '#BC8F8F', 'royalblue': '#4169E1',
                'saddlebrown': '#8B4513', 'sage': '#87AE73',
                'salmon': '#FA8072', 'sandybrown': '#FAA460',
                'seagreen': '#2E8B57', 'seashell': '#FFF5EE',
                'sienna': '#A0522D', 'silver': '#C0C0C0',
                'skyblue': '#87CEEB', 'slateblue': '#6A5ACD',
                'slategray': '#708090', 'slategrey': '#708090',
                'snow': '#FFFAFA', 'springgreen': '#00FF7F',
                'steelblue': '#4682B4', 'tan': '#D2B48C',
                'teal': '#008080', 'thistle': '#D8BFD8',
                'tomato': '#FF6347', 'turquoise': '#40E0D0',
                'violet': '#EE82EE', 'wheat': '#F5DEB3',
                'white': '#FFFFFF', 'whitesmoke': '#F5F5F5',
                'yellow': '#FFFF00', 'yellowgreen': '#9ACD32'}

