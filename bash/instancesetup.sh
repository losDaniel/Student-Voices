# To be run within an instance 
# Make sure the folder structure, directories and required libraries are present 

# The EFS directory should have been created when mounting the storage onto the instance during launch 
cd efs

# Use "sudo update-alternatives --config python" in shell to see available python configs
# Change the default python version to use as ptyhon 3.6 
sudo update-alternatives --set python /usr/bin/python3.6

# Upgrade pip 
pip install --upgrade pip 
pip install TextBlob
pip install nltk 
pip install gensim
pip install sklearn
pip install pandas

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

python -m nltk.downloader all
sudo python -m nltk.downloader -d /usr/local/share/nltk_data all