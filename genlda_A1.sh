# To be run within an instance 
# Run the LDA Analysis: 
# -- create the directories you need 
# -- pull scripts directory from github 
# -- run data cleaning 

cd efs
mkdir scripts
cd scripts
git init
git config core.sparsecheckout true
echo scripts/ >> .git/info/sparse-checkout
git remote add -f origin https://github.com/losDaniel/Student-Voices.git
git checkout cleaning_script
git pull origin
cd scripts 
python modeling_tools.py
python lda_analysis.py
nohup sudo python analysis_a2_clean_data.py -c A1 &> clean_A1.txt &
curpid=$!
nohup sh -c 'while ps -p $0 > /dev/null; do sleep 10; done && sudo shutdown -h now' $curpid