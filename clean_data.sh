pip install --upgrade pip 
pip install TextBlob
cd efs
mkdir clean_A1
mkdir clean_A2
mkdir clean_A3
cd clean_A1
git init
git config core.sparsecheckout true
echo python/** >> .git/info/sparse-checkout
git remote add -f origin https://github.com/losDaniel/Student-Voices.git
git checkout cleaning_script
python modeling_tools.py