# Student-Voices

This repository holds the work that is the basis of an on-going series of publications studying the perception of teacher quality in the United States. We scraped millions of teacher reviews submitted by students and parents using the natural language processing methods to retrieve trends in how students perceive and describe their teachers and classes across the country and over time.   

While the most of the work is compelte, it still needs to be reviewed before it can be made public, and we only focus on the parts relevant to each new publication. Thus, this repository will be updated as we complete each new paper. 

## The Subjects of Bad Reviews Using LDA

In the first commit to this repo we uplodaed an analysis of the reviews that rated teachers from 0 to 35 out of 100. This sub-corpus included 359,387 total reviews. We experimented with a number of different cleaning processes and model parameters, before settling on 4 viable candidate parameters for an in depth experimental trial. Some critical findings in early stage tweeking was the importance of setting a minimum review length and incorporating bigram & ngram phrase detection.

To analyze these reviews we experimented using LDA models available through the [`gensim`](https://radimrehurek.com/gensim/index.html) python package. We created a number of personalized visualizations, we also created visualizations using the [`pyLDAvis`](https://pypi.org/project/pyLDAvis/) plug-in for gensim. Below is an example visualization of the topics (numbered bubbles) and the themes they were grouped into (larger colored bubbles). The results from this analysis are summarized in the "General Conclusions" section of the `(B4) lda_results_0-35` notebook (needs to be updated to align with publication). 

![LDA Topic Visualization](distance_map.png)

## The Scraper 

**Source Update:** The scraper was designed to work on the website RateMyTeacher.com when its terms of use and formatting allowed for collection of their public data. As of the summer of 2019 the website underwent a redesign which has rendered much (if not all) of the text in the reviews inaccessible. Thus, this scraper does not work on the current version of the website. Nevertheless, we continue to make the database compiled through our original scrape available for use in analysis. This contains the millions of reviews submitted over the span of more than a decade in six english speaking countries around the world.

The `rmt_scraper` class is designed to scrape millions of reviews submitted from almost every english speaking country in the world. It is capable of executing multi-threaded scraping and uses a combination of selenium and beautiful soup to: 

1) navigate each geographic level collecting the relevant links for each geography contained within from country to school. 
2) scrape school level information and teacher specific ratings links from each school page. 
3) scrape teacher descriptions and reviews from each teacher link. 

Each review scraped included the text of the review submitted, likert scale responses for a number of qualities, and a numeric rating from 1 to 5 (in steps of 0.5) which we later scaled from 0 to 100. 

The scraper can be executed from notebook `(A1) scrape_database`. Basic and detailed summary statistics can be found in notebooks `(A2) explore_clean_setup` and `(A3) teacher_stats` respectively. We used [`git large file storage`](https://git-lfs.github.com/) to store larger files. The number of reviews changes over time, at the time of our scraping we collected 4,863,978
reviews. 

## Notebooks 

The notebooks can be used to run the different stages of the project. Notation: the letter indicates the phase of the project that is being run (i.e. data collection, LDA analysis, language modeling, etc...) and the number indicates the dependency level within the phase. This is only true across phases if the numbering of a given phase does not include earlier numbers. Thus, A1 generates materials necessary for A2, A2 -> A3 & B3, B3 -> B4, but A3 is not needed for B4.  

**(A1) scrape_database.ipynb**: scrapes all or a subset of teachers reviews and stores them in a series of pickle files. <br><br>
**(A2) explore_clean_setup.ipynb**: imports, appends, splits the data into quantitative and text datasets, generates different cleaned versions of the text, provides summary statistics.<br><br>
**(A3) teacher_stats.ipynb**: a more in depth look at the quantitative side of the data. <br><br>
**(B3) lda_topics.ipynb**: initial exploration of potential LDA experimentation strategies and execution of a 4-trial lda experiment. <br><br>
**(B4) lda_results_0-35.ipynb**: reviews of LDA outcomes for the 0-35 rated sub-corpus of the entire body of reviews. (Note: this notebook will not render properly on github, to view it fully rendered visit it using nbviewer [here](https://nbviewer.jupyter.org/github/dankundertone/Student-Voices/blob/master/%28B4%29%20lda_results_0-35.ipynb)) <br><br>

### Dependencies 

`pip install pandas`<br>
`pip install numpy`<br>
`pip install bs4` # BeautifulSoup<br>
`pip install selenium` # Download the [chromewebdriver](https://chromedriver.chromium.org/downloads)<br>
`pip install nltk`<br>
`pip install gensim`<br>
`pip install sklearn`<br>
