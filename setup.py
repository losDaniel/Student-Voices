from setuptools import setup, find_packages 
from os import path 

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f: 
	long_description = f.read()

setup( 
	name='student-voices',
	version='0.0.1',
	description='Project module for review of RMT study.',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/losDaniel/Student-Voices.git',
	author='Carlos Valcarcel',
	author_email='losdaniel@berkeley.edu',
	license='unlicensed',
	keywords='nlp education reviews',
	packages=['student_voices'],
	entry_points={
		'console_scripts':[
			# 'create_levels = day_trader.create_lvl_output:main',
			# 'run_apr = day_trader.run_apr:main',
			# 'run_trader = day_trader.run_trader:main'
		]
	},
	install_requires=['pandas','nltk','gensim','sklearn','TextBlob', 'spot-connect', 'pyLDAvis'], #selenium and bs4 are excluded because scraper is no longer relevant 
	package_data={
		'student_voices':[
			# 'data/current_session_ids.pickle',
			# 'data/daylight_savings.pickle',
		]
	},
	include_package_data=True,
)