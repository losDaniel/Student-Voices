# Student-Voices

This is the project repository for: 

*Valcarcel, C. Holmes, J. Berliner, D. Koerner, M. (2021). The Value of Student Feedback in Open Forums: A Natural Analysis of Descriptions of Poorly-Rated Teachers. Education Policy Analysis Archives, 26(number).* 

Graphs and other results can be found in their respective directories. For this analysis we scraped millions of teacher reviews submitted by students and parents and used  natural language processing to retrieve patterns in how students describe their teachers and classes across the U.S. 

## The Scraper 

**Source Update:** *The scraper does not work on the current version of the website.* The scraper was designed to work on the website RateMyTeacher.com when its terms of use and formatting allowed for collection of their public data. As of the summer of 2019 the website underwent a redesign which has rendered much (if not all) of the text in the reviews inaccessible. We do not know if the website preserved the data from other English speaking countries. We only have data for the U.S. We continue to make the database available. 

**Download Text Dataset**: [download link](https://www.filefactory.com/file/5h8slx7a527y/full_review_text.pbz2). 

This contains the millions of reviews submitted over the span of more than a decade in the U.S. Each review scraped included the text of the review submitted, likert scale responses for a number of qualities, and a numeric rating from 1 to 5 (in steps of 0.5) which we later scaled from 0 to 100. 

## Notebooks 

The notebooks are ordered, if executed in order one should be able to replicate most of the steps in our analysis. 

The repository is set up as a package so you can download the zip, unzip it, navigate to the directory and use `pip install -e .` to install the `student_voices` package necessary to run the notebooks. 