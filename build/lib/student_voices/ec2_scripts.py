import os, sys
from path import Path
root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root.parent)


def get_instance_setup_script():
    
    # To be run within an instance 
    # Make sure the folder structure, directories and required libraries are present 
    
    script = ''

    # The EFS directory should have been created when mounting the storage onto the instance during launch 
    script+= 'cd efs\n'
    
    # Use "sudo update-alternatives --config python" in shell to see available python configs
    # Change the default python version to use as ptyhon 3.6 
    script+= 'sudo update-alternatives --set python /usr/bin/python3.6\n'
    
    # Upgrade pip 
    script+= 'pip install --upgrade pip \n'
    script+= 'pip install TextBlob\n'
    script+= 'pip install nltk\n'
    script+= 'pip install gensim\n'
    script+= 'pip install sklearn\n'
    script+= 'pip install pandas\n'
    
    # Create the directories and subdirectories we needs 
    script+= 'mkdir results\n'
    
    script+= 'cd results\n'
    script+= 'mkdir LDAdescriptions\n'
    script+= 'mkdir LDAdistributions\n'
    
    script+= 'cd ..\n'
    script+= 'mkdir models\n'
    
    script+= 'mkdir graphs\n'
    script+= 'cd graphs\n'
    script+= 'mkdir LDAGraphs\n'
    
    script+= 'python -m nltk.downloader all\n'
    script+= 'sudo python -m nltk.downloader -d /usr/local/share/nltk_data all\n'
    
    return script 


def get_clean_data_script(config, log_file_name, overwrite=True):
    # To be run within an instance 
    # Run the data cleaning: 
    # -- create the directories you need 
    # -- pull scripts directory from github 
    # -- run data cleaning 
    
    script = ''
    
    script+= 'cd efs\n' 
    
    # Run the modeling tools script for a preliminary setup 
    script+= 'python /Student_Voices/student_voices/modeling_tools.py\n'
        
    # Rigged to run with notebook
    script+= 'nohup python /Student_Voices/student_voices/clean_data.py -c '+str(config)+' -o '+str(overwrite)+' &> '+str(log_file_name)+' &\n'
    script+= 'curpid=$!\n'
    
    # Wait until the previous job is done and then shutdown the instance 
    script+= "nohup sh -c 'while ps -p $0 > /dev/null; do sleep 10; done && sudo shutdown -h now' $curpid &> run.txt\n"

    return script     


def get_spot_lda_script(data_config, setting):
    
    # To be run within an instance 
    # Run the LDA Analysis: 
    # -- create the directories you need 
    # -- pull scripts directory from github 
    # -- run data cleaning 
    
    script = ''
    
    # Create the necessary directories if they're not around
    script+='cd efs\n'
    script+='mkdir scripts\n'
    script+='cd scripts\n'
    
    # Pull the scripts from github
    script+='git init\n'
    script+='git config core.sparsecheckout true\n'
    script+='echo scripts/ >> .git/info/sparse-checkout\n'
    script+='git remote add -f origin https://github.com/losDaniel/Student-Voices.git\n'
    script+='git checkout cleaning_script\n'
    script+='git pull origin\n'
    script+='cd scripts\n'
    
    # Run the modeling tools script for a preliminary setup 
    script+='python modeling_tools.py\n'
    script+='python lda_analysis.py\n'
    
    # Pull the arguments to pass to the command and save the log 
    script+='CON=$1\n'
    script+='SET=$2\n'
    script+='TXTFILE="LDA_$CON_$SET.txt"\n'
    script+='nohup sudo python lda_modeling.py -c $CON -s $SET &> $TXTFILE &\n'
    script+='curpid=$!\n'
    
    # Wait until the previous job is done and then shutdown the instance 
    script+="nohup sh -c 'while ps -p $0 > /dev/null; do sleep 10; done && sudo shutdown -h now' $curpid\n"
    
    return script 
    