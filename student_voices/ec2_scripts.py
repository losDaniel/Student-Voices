import os, sys
from path import Path
root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root.parent)


def get_instance_setup_script(run_as_user='', delimiter='\n'):
    
    # To be run within an instance 
    # Make sure the folder structure, directories and required libraries are present 
    
    script = ''

    # The EFS directory should have been created when mounting the storage onto the instance during launch 
    script+= 'cd efs'+delimiter
    
    # Use "sudo update-alternatives --config python" in shell to see available python configs
    # Change the default python version to use as ptyhon 3.6 
    script+= 'sudo update-alternatives --set python /usr/bin/python3.6\n'+delimiter

    if run_as_user =='': 
        script += 'pip install -e /home/'+run_as_user+'/efs/Student_Voices/'+delimiter
    else: 
        # Even though we don't need the sudo command in user_data, we use it in case this method were used out of the root command line
        script += 'sudo runuser -l '+run_as_user+" -c 'pip install -e /home/"+run_as_user+"/efs/Student_Voices/'"+delimiter
    
#    # Upgrade pip 
#    script+= 'pip install --upgrade pip \n'
#    script+= 'pip install TextBlob\n'
#    script+= 'pip install nltk\n'
#    script+= 'pip install gensim\n'
#    script+= 'pip install sklearn\n'
#    script+= 'pip install pandas\n'
    
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


def get_clean_data_script(config, log_file_name, overwrite=True, delimiter='\n'):
    # To be run within an instance 
    # Run the data cleaning: 
    # -- create the directories you need 
    # -- pull scripts directory from github 
    # -- run data cleaning 
    
    script = ''
    
    script+= 'cd efs'+delimiter 
    
    # Run the modeling tools script for a preliminary setup 
    script+= 'python /Student_Voices/student_voices/modeling_tools.py'+delimiter
        
    # Rigged to run with notebook
    script+= 'nohup python /Student_Voices/student_voices/clean_data.py -c '+str(config)+' -o '+str(overwrite)+' &> '+str(log_file_name)+' &'+delimiter
    script+= 'curpid=$!'+delimiter
    
    # Wait until the previous job is done and then shutdown the instance 
    script+= "nohup sh -c 'while ps -p $0 > /dev/null; do sleep 10; done && sudo shutdown -h now' $curpid &> run.txt"+delimiter

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
    



import sys
sys.path.append('../')

from spot_connect import bash_scripts, efs_methods, instance_manager

# Use a function to create the instance-specific parts of each user_data script 
def produce_level_creation_script(work, mof, lpm, filepath, logname='', region='us-east-2', run_as_user='', delimiter='\n', script=''):
    
    if run_as_user =='': 
        script += 'pip install -e /home/'+run_as_user+'/efs/Day-Trader/'+delimiter
    else: 
        # Even though we don't need the sudo command in user_data, we use it in case this method were used out of the root command line
        script += 'sudo runuser -l '+run_as_user+" -c 'pip install -e /home/"+run_as_user+"/efs/Day-Trader/'"+delimiter
    
    # Compose the arguments 
    level_path = '/home/ec2-user/efs/modified_data/output_levels/'    
    
    arg = '-sd '+str(work)+' -lp '+level_path+' -mof '+str(mof)+' -lpm '+str(lpm)+' -fp '+filepath
    
    command = 'create_levels '+arg
    
    # Use the package commands to run the job 
    if run_as_user=='': 
        script = bash_scripts.cancel_fleet_after_command(command, region, command_log=logname, script=script)
    else: 
        script = bash_scripts.cancel_fleet_after_command(command, region, command_log=logname, run_as_user=run_as_user, script=script)
    
    return script


def produce_model_training_script(model_script, model_params, cancel_fleet=True, logname='', region='us-east-2', run_as_user='', delimiter='\n', script=''):

    if run_as_user =='': 
        script += 'pip install -e /home/'+run_as_user+'/efs/Day-Trader/'+delimiter
    else: 
        # Even though we don't need the sudo command in user_data, we use it in case this method were used out of the root command line
        script += 'sudo runuser -l '+run_as_user+" -c 'pip install -e /home/"+run_as_user+"/efs/Day-Trader/'"+delimiter
    
    # DEV - check for log file name 
    script += 'echo "-----------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"'
    script += 'echo "'+logname+'"'
    script += 'echo "-----------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"'
    
    command = 'python '+str(model_script)+' -mn '+str(model_params['models_name'])+' -nn '+str(model_params['models_nickname'])+' -apr '+str(model_params['is_apr'])+' -asp '+str(model_params['apr_sample_path'])+' -mdp '+str(model_params['modified_data_path'])+' -dbp '+str(model_params['database_path'])+' -mpath '+str(model_params['models_path'])+' -ep '+str(model_params['epochs'])+' -ptt '+str(model_params['percent_to_train'])
    if model_params['transfer'] is not None: 
        command += ' -tr '+str(model_params['transfer'])
    
    if cancel_fleet: 
        if run_as_user=='':
            script = bash_scripts.cancel_fleet_after_command(command, region, command_log=logname, script=script)
        else: 
            script = bash_scripts.cancel_fleet_after_command(command, region, command_log=logname, run_as_user=run_as_user, script=script)
    else: 
        if run_as_user=='': 
            script += command+logname+delimiter
        else: 
            script +=bash_scripts.run_command_as_user(command, user=run_as_user, delimiter=delimiter)
        
    return script 


def create_training_user_data_script(model_params, filesystem, region, cancel_fleet, availability_zone=None):

    # Bash scripts submitted as user data must have the correct header 
    working_script = bash_scripts.init_userdata_script()
    
    # If an availability zone is submitted use the mount target formula to generate a mount dns  
    if availability_zone is not None: 
        mount_dns = efs_methods.get_mounttarget_dns(filesystem, region, availability_zone)    
    # If no availability zone is submitted then use the filesystem dns to mount the instance 
    else: 
        mount_dns = efs_methods.get_filesystem_dns(filesystem, region=region)

    working_script = bash_scripts.compose_mount_script(
        mount_dns, 
        script=working_script
    )

    # Installs 
    working_script = bash_scripts.install_ta_lib(
        install_as_user='ec2-user', 
        script=working_script
    )

    working_script += bash_scripts.run_command_as_user(
            'pip install spot-connect',
            'ec2-user', 
            delimiter='\n')

    if cancel_fleet=='False':
        cancel_fleet = False 
    elif cancel_fleet=='True': 
        cancel_fleet = True 
    else: 
        raise TypeError('cancel fleet param must be "True" or "False"')

    log_file_name = model_params['models_path']+'/'+model_params['models_name']+'/'+model_params['models_nickname']+'/training_'+model_params['models_name']+'_'+model_params['models_nickname']+'.txt'

    model_script = model_params['model_script_dir']+'/'+model_params['models_name']+'_'+model_params['models_nickname']+'.py'

    working_script=produce_model_training_script(model_script,
                                                 model_params,
                                                 cancel_fleet=cancel_fleet,
                                                 logname=log_file_name,
                                                 region=region,
                                                 run_as_user='ec2-user',
                                                 script=working_script)
    
    # Convert the working script to base-64 encoded so the fleet can run it 
    user_data_script = bash_scripts.script_to_userdata(working_script)

    return user_data_script


def launch_fleet_training(model_params, prefix, instance_type, filesystem, region, availability_zone, cancel_fleet, instance_profile, account_number_file):
    
    # Create a user data script to launch the specified training job     
    user_data_script = create_training_user_data_script(model_params, filesystem, region, cancel_fleet)

    # Use an instance manager to create the fleet and distribute the job 
    im = instance_manager.InstanceManager()
    try: 
        account_num = open(account_number_file).read()
    except: 
        raise Exception('Please replace the path above with a .txt file that contains you account number')
        
    im.run_distributed_jobs(account_num, prefix, 1, instance_type, availability_zone=availability_zone, user_data=[user_data_script], instance_profile=instance_profile)

