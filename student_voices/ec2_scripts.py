import os, sys
from path import Path
root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root.parent)

from spot_connect import bash_scripts, efs_methods


def get_instance_setup_script(filesystem, region, availability_zone=None, run_as_user='', delimiter='\n'):
    
    # To be run within an instance 
    # Make sure the folder structure, directories and required libraries are present 
    # Bash scripts submitted as user data must have the correct header 
    script = bash_scripts.init_userdata_script()
    
    # If an availability zone is submitted use the mount target formula to generate a mount dns  
    if availability_zone is not None: 
        mount_dns = efs_methods.get_mounttarget_dns(filesystem, region, availability_zone)    
    # If no availability zone is submitted then use the filesystem dns to mount the instance 
    else: 
        mount_dns = efs_methods.get_filesystem_dns(filesystem, region=region)

    script = bash_scripts.compose_mount_script(mount_dns, script=script)

    # The EFS directory should have been created when mounting the storage onto the instance during launch 
    script+= 'cd efs'+delimiter
    
    # Use "sudo update-alternatives --config python" in shell to see available python configs
    # Change the default python version to use as ptyhon 3.6 

    if run_as_user =='': 
        script += 'sudo update-alternatives --set python /usr/bin/python3.6'+delimiter
        script += 'pip install -e /home/'+run_as_user+'/efs/Student-Voices/'+delimiter
    else: 
        script += 'sudo runuser -l '+run_as_user+" -c 'sudo update-alternatives --set python /usr/bin/python3.6'"+delimiter
        # Even though we don't need the sudo command in user_data, we use it in case this method were used out of the root command line
        script += 'sudo runuser -l '+run_as_user+" -c 'pip install -e /home/"+run_as_user+"/efs/Student-Voices/'"+delimiter
        
    # Create the directories and subdirectories we needs 
    script+= 'mkdir results'+delimiter
    
    script+= 'cd results'+delimiter
    script+= 'mkdir LDAdescriptions'+delimiter
    script+= 'mkdir LDAdistributions'+delimiter
    
    script+= 'cd ..'+delimiter
    script+= 'mkdir models'+delimiter
    
    script+= 'mkdir graphs'+delimiter
    script+= 'cd graphs'+delimiter
    script+= 'mkdir LDAGraphs'+delimiter

    if run_as_user =='': 
        script+= 'python -m nltk.downloader all'+delimiter
#        script+= 'sudo python -m nltk.downloader -d /usr/local/share/nltk_data all'+delimiter
    else: 
        # Even though we don't need the sudo command in user_data, we use it in case this method were used out of the root command line
        script += 'sudo runuser -l '+run_as_user+" -c 'python -m nltk.downloader all'"+delimiter
#        script += 'sudo runuser -l '+run_as_user+" -c 'python -m nltk.downloader -d /usr/local/share/nltk_data all'"+delimiter

    return script 



def get_clean_data_script(config, log_file_name, region, path, cancel_fleet=True, overwrite=True, script='', run_as_user='', delimiter='\n'):
    # To be run within an instance 
    # Run the data cleaning: 
    # -- create the directories you need 
    # -- pull scripts directory from github 
    # -- run data cleaning 
    
    if script=='':    
        # Bash scripts submitted as user data must have the correct header 
        script = bash_scripts.init_userdata_script()
    
    script+= 'cd efs'+delimiter 
    
    # Use the package commands to run the job 
    if run_as_user=='': 
        # Run the modeling tools script for a preliminary setup 
        script+= 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'+delimiter
    else: 
        script += 'sudo runuser -l '+run_as_user+" -c 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'"+delimiter

    command = 'python /home/ec2-user/efs/Student-Voices/student_voices/clean_data.py -c '+str(config)+' -o '+str(overwrite)+' -p '+str(path) 
        
    if cancel_fleet: 
        # Use the package commands to run the job 
        if run_as_user=='': 
            script = bash_scripts.cancel_fleet_after_command(command, region, command_log=log_file_name, script=script)
        else: 
            script = bash_scripts.cancel_fleet_after_command(command, region, command_log=log_file_name, run_as_user=run_as_user, script=script)
    else: 
        if run_as_user=='': 
            script += command+'> '+log_file_name+delimiter
        else: 
            command += '> '+log_file_name
            script +=bash_scripts.run_command_as_user(command, user=run_as_user, delimiter=delimiter)    

    return script     


def get_lda_script(config, setting, model_dir, config_path, log_file_name, region, cancel_fleet=True, script='', run_as_user='', delimiter='\n'):
    # To be run within an instance 
    
    if script=='':    
        # Bash scripts submitted as user data must have the correct header 
        script = bash_scripts.init_userdata_script()
    
    script+= 'cd /home/ec2-user/efs'+delimiter 
    
    script+= 'mkdir '+model_dir+'/'+setting+'_'+config
    
    # Use the package commands to run the job 
    if run_as_user=='': 
        # Run the modeling tools script for a preliminary setup 
        script+= 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'+delimiter
    else: 
        script += 'sudo runuser -l '+run_as_user+" -c 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'"+delimiter

    command = 'python /home/ec2-user/efs/Student-Voices/student_voices/run_lda.py -c '+str(config)+' -cp '+str(config_path)+' -md '+str(model_dir)+' -s '+str(setting) 
        
    if cancel_fleet: 
        # Use the package commands to run the job 
        if run_as_user=='': 
            script = bash_scripts.cancel_fleet_after_command(command, region, command_log=log_file_name, script=script)
        else: 
            script = bash_scripts.cancel_fleet_after_command(command, region, command_log=log_file_name, run_as_user=run_as_user, script=script)
    else: 
        if run_as_user=='': 
            script += command+'> '+log_file_name+delimiter
        else: 
            command += '> '+log_file_name
            script +=bash_scripts.run_command_as_user(command, user=run_as_user, delimiter=delimiter)    

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
    


