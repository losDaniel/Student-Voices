# To be run within an instance 
# Setup the instance for data cleaning: 
# -- create the necessary directories if they are not present 
# -- install TextBlob and nltk 
# -- run nltk.download() through bash  

cd efs
pip install --upgrade pip 
mkdir scripts 
mkdir data
mkdir results
cd results 
mkdir LDAdescriptions
mkdir LDAdistributions
cd .. 
mkdir models
mkdir graphs 
cd graphs 
mkdir LDAGraphs
cd ..
pip install TextBlob
pip install nltk 
python -m nltk.downloader all
sudo python -m nltk.downloader -d /usr/local/share/nltk_data all