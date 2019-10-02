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

cd ..

mkdir scripts 

cd scripts
# Pull the scripts from github
git init
git config core.sparsecheckout true
echo scripts/ >> .git/info/sparse-checkout
git remote add -f origin https://github.com/losDaniel/Student-Voices.git
git checkout cleaning_script
git pull origin



python -m nltk.downloader all
sudo python -m nltk.downloader -d /usr/local/share/nltk_data all


# Install Text Blob
pip install TextBlob
# Install NLTK
pip install nltk 

# Create the necessary directories if they're not around
cd efs
mkdir scripts
cd scripts 
