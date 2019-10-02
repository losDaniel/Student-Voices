# To be run within an instance 
# Run the data cleaning: 
# -- create the directories you need 
# -- pull scripts directory from github 
# -- run data cleaning 

# Install Text Blob
pip install TextBlob
# Install NLTK
pip install nltk 

# Create the necessary directories if they're not around
cd efs
mkdir scripts
cd scripts

# Pull the scripts from github
git init
git config core.sparsecheckout true
echo scripts/ >> .git/info/sparse-checkout
git remote add -f origin https://github.com/losDaniel/Student-Voices.git
git checkout cleaning_script
git pull origin
cd scripts 

# Run the modeling tools script for a preliminary setup 
python modeling_tools.py

# Pull the arguments to pass to the command and save the log 
ARG=$1
TXTFILE="clean_$ARG.txt"
nohup python cleaning.py -c $ARG -o True &> $TXTFILE &
curpid=$!

# Wait until the previous job is done and then shutdown the instance 
nohup sh -c 'while ps -p $0 > /dev/null; do sleep 10; done && sudo shutdown -h now' $curpid