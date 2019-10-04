# Connect to the instance and update scripts directory using git pull

# Make your way to the scripts directory 
cd efs
mkdir scripts 
cd scripts

# Pull the scripts from github
git init
git config core.sparsecheckout true
echo scripts/ >> .git/info/sparse-checkout
git remote add -f origin https://github.com/losDaniel/Student-Voices.git

# Select the branch you want to update with 
git checkout cleaning_script
git pull origin