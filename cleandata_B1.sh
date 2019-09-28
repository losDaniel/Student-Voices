pip install --upgrade pip 
pip install TextBlob
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
nohup python analysis_a2_clean_data.py -c B1 > clean_B1.txt &
sudo shutdown -h now