# To be run within an instance 
# Make sure the folder structure, directories and required libraries are present 

# The EFS directory should have been created when mounting the storage onto the instance during launch 
cd efs

# Upgrade pip 
pip install --upgrade pip 

# Create the directories and subdirectories we needs 
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

# Install Text Blob
pip install TextBlob
# Install NLTK
pip install nltk 

python -m nltk.downloader all
sudo python -m nltk.downloader -d /usr/local/share/nltk_data all