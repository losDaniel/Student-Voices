# Student-Voices

This is the project repository for: 

Valcarcel, C. Holmes, J. Berliner, D.C. Koerner, M. (2021). *The Value of Student Feedback in Open Forums: A Natural Analysis of Descriptions of Poorly-Rated Teachers.* Education Policy Analysis Archives, 26(number). 

Graphs and other results can be found in their respective directories. For this analysis we scraped millions of teacher reviews submitted by students and parents and used  natural language processing to retrieve patterns in how students describe their teachers and classes across the U.S. The graph below shows the reviews, clustered by a 20-topic LDA model and plotted in two dimensions. They are color coded according the general theme associations found in a thematic analysis of sample reviews.

![color plot](https://github.com/losDaniel/Student-Voices/blob/master/graphs/Thematic%202d%20Plot%20Color.png)

## Data & The Scraper 

**Source Update:** *The scraper does not work on the current version of the website.* The scraper was designed to work on the website RateMyTeacher.com when its terms of use and formatting allowed for collection of their public data. As of the summer of 2019 the website underwent a redesign which has rendered much (if not all) of the text in the reviews inaccessible. We do not know if the website preserved the data from other English speaking countries. We only have data for the U.S. We continue to make the database available. 

**Download Text Dataset**: [download link](https://www.filefactory.com/file/5h8slx7a527y/full_review_text.pbz2). 

This contains the millions of reviews submitted over the span of more than a decade in the U.S. Each review scraped included the text of the review submitted, likert scale responses for a number of qualities, and a numeric ratings. There are also likert scale responses that rate things like text book use and other things directly. 

## Notebooks 

The notebooks are ordered, if executed in order one should be able to replicate most of the steps in our analysis. 

The repository is set up as a package so you can download the zip, unzip it, navigate to the directory and use `pip install -e .` to install the `student_voices` package necessary to run the notebooks. 

## Future Research 

There is still a lot of potential to check the correlation between topics yielded from NLP and likert scale responses. This could reveal something about the accuracy of certain likert scale options which could be used to improve conventional feedback mechanisms in schools. Further research using other segments of the data than those we analyzed in the paper above would also be of interest.