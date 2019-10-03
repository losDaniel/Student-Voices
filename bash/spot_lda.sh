# To be run within an instance 
# Run the LDA Analysis: 
# -- create the directories you need 
# -- pull scripts directory from github 
# -- run data cleaning 

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
python lda_analysis.py

# Pull the arguments to pass to the command and save the log 
CON=$1
SET=$2
TXTFILE="LDA_$CON_$SET.txt"
nohup sudo python lda_modeling.py -c $CON -s $SET &> $TXTFILE &
curpid=$!

# Wait until the previous job is done and then shutdown the instance 
nohup sh -c 'while ps -p $0 > /dev/null; do sleep 10; done && sudo shutdown -h now' $curpid