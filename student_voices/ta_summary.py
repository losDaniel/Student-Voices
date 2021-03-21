import numpy as np
import pandas as pd


def clean_db(db):
    new_db = {}
    for i, sheet in enumerate(list(db.keys())):
        
        df = db[sheet]
        subject_columns = [c for c in df.columns if (c!='Text') & (':' not in c)]

        df.columns = [c.strip() for c in df.columns]
        subject_columns = [c.strip() for c in subject_columns]

        df['rowsum'] = df[subject_columns].sum(axis=1)
        df['rowsum'] = np.where(df['rowsum']==0, np.nan, df['rowsum'])
        df['rowsum'] = df['rowsum'].bfill()
        coded_rows = df[df['rowsum'].notnull()]

        coded_rows = coded_rows.drop('rowsum', 1)

        new_db[sheet] = coded_rows 
        
    return new_db 

def create_basic_summary(db):
    '''Takes a thematic analysis excel file and returns a summary for each sheet'''
    basic_summary = {} 

    for i, sheet in enumerate(list(db.keys())): 
        df = db[sheet]

        subject_columns = [c for c in df.columns if (c!='Text') & (':' not in c)]

        basic_dist = df[subject_columns].sum(axis=0)/len(df)
        basic_dist = pd.DataFrame(basic_dist, columns=['p']).reset_index()
        basic_dist = basic_dist[basic_dist['p']!=0]
        basic_dist.columns = ['theme','p']

        basic_summary[sheet] = (basic_dist, subject_columns) 

    return basic_summary 

def create_main_summary(db): 
	'''Takes a thematic analysis and returns a summary for each sheet with general summary'''
	main_summary = {} 

	for i, sheet in enumerate(list(db.keys())):
		df = db[sheet]

		subject_columns = [c for c in df.columns if (c!='Text') & ((':' not in c) or ('G:' in c))]

		main_dist = df[subject_columns].sum(axis=0)/len(df)
		main_summary[sheet] = main_dist

	return main_summary 


def check_themes_used(basic_summary, db):
	'''Check the themes that weren't used, some of these may be larger categories so don't automatically remove all of them'''
	themes_used = []
	for sheet in basic_summary:
	    themes_used += list(basic_summary[sheet][0]['theme'])
	themes_used = list(set(themes_used))
	subject_columns = [c for c in db[sheet].columns if (c!='Text') & (':' not in c)]
	themes_not_used = [c for c in subject_columns if c not in themes_used]

	return themes_used, themes_not_used


def add_general_themes(db):
	'''Add variables for general (aggregated) themes'''
	new_db = {} 
	try: 
		for sheet in db:
			new_db[sheet] = tabulate_general_themes(db[sheet])
	except Exception as e: 
		print(sheet)
		raise e
	return db

def tabulate_general_themes(df):
	subjects = ['subjects',
	 'drama',
	 'gym',
	 'band',
	 'language barrier',
	 'math']

	bad_relationship = [ 'bad relationship',
	 'parent rel',
	 'past / gpa / unprepared']

	personality = ['personality',
	 'biased',
	 'picks on kids',
	 'picks favorites',
	 'does not care / lazy',
	 'gets angry',
	 'at questions',
	 'mean / no respect',
	 'scary',
	 'yells',
	 'unprofesional',
	 'annoying / frustrating / abrasive',
	 'cool / nice',
	 'treat like kids / fools',
	 'strict']

	df['G:PS'] = 0 
	for c in personality:
		df['G:PS'] = np.where(df[c]==1,1,df['G:PS'])

	df['G:PS/BR'] = 0 
	for c in personality + bad_relationship:
		df['G:PS/BR'] = np.where(df[c]==1,1,df['G:PS/BR'])

	duties = [ 
	 'duties',
	 'lack expertise',
	 'makes mistakes',
	 'distracted',
	 'does other things',
	 'chaos in classroom',
	 'disorganized',
	 'slow / no grades',
	 'no follow-through']

	df['G:DT'] = 0 
	for c in duties:
		df['G:DT'] = np.where(df[c]==1,1,df['G:DT'])

	no_learning = [
	 'no learning',
	 'struggling in follow up',
	 'past / gpa / unprepared']

	not_teaching = ['boring',
	 'busy work',
	 'waste / pointless',
	 'cannot explain / unclear',
	 'confusing',
	 'rushed',
	 'unhelpful',
	 'ignores students / questions',
	 'discouraging / not motivating',
	 'off-topic',
		 'by the book / ppt']

	workload = ['unrealistic expectations',
	 'too much work',
	 'hard / unfair (grader)',
	 'bad feedback',
	 'unclear direction',
	 'study guides',
	 'misaligned',
	 'easy a']


	df['G:WL'] = 0 
	for c in workload:
		df['G:WL'] = np.where(df[c]==1,1,df['G:WL'])

	df['G:NT'] = 0 
	for c in not_teaching: 
		df['G:NT'] = np.where(df[c]==1,1,df['G:NT'])

	df['G:NT/NL'] = 0 
	for c in not_teaching + no_learning:
		df['G:NT/NL'] = np.where(df[c]==1,1,df['G:NT/NL'])

	df['G:NT/NL/WL'] = 0 
	for c in not_teaching + no_learning + workload:
		df['G:NT/NL/WL'] = np.where(df[c]==1,1,df['G:NT/NL/WL'])


	return df
