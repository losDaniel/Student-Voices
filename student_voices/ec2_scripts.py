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
        
#    # Create the directories and subdirectories we needs 
#    script+= 'mkdir results'+delimiter
#    
#    script+= 'cd results'+delimiter
#    script+= 'mkdir LDAdescriptions'+delimiter
#    script+= 'mkdir LDAdistributions'+delimiter
#    
#    script+= 'cd ..'+delimiter
#    script+= 'mkdir models'+delimiter
#    
#    script+= 'mkdir graphs'+delimiter
#    script+= 'cd graphs'+delimiter
#    script+= 'mkdir LDAGraphs'+delimiter

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


def get_lda_script(config, setting, ntop, corpus_group, model_dir, config_path, log_file_name, region, cancel_fleet=True, script='', run_as_user='', delimiter='\n'):
    # To be run within an instance 
    
    if script=='':    
        # Bash scripts submitted as user data must have the correct header 
        script = bash_scripts.init_userdata_script()
    
    script+= 'cd /home/ec2-user/efs/models'+delimiter 
    
    #script+= 'mkdir '+model_dir+'/'+setting+'_'+config+delimiter
    
    # Use the package commands to run the job 
    if run_as_user=='': 
        # Run the modeling tools script for a preliminary setup 
        script+= 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'+delimiter
    else: 
        script += 'sudo runuser -l '+run_as_user+" -c 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'"+delimiter

    command = 'python /home/ec2-user/efs/Student-Voices/student_voices/run_lda.py -c '+str(config)+' -cp '+str(config_path)+' -md '+str(model_dir)+' -s '+str(setting)+' -nt '+str(ntop)+' -cg '+str(corpus_group)
        
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



def get_coherence_script(config, setting, ntop, corpus_group, model_dir, config_path, results_path, log_file_name, region, cancel_fleet=True, script='', run_as_user='', delimiter='\n'):
    # To be run within an instance 
    
    if script=='':    
        # Bash scripts submitted as user data must have the correct header 
        script = bash_scripts.init_userdata_script()
    
    script+= 'cd /home/ec2-user/efs/results'+delimiter 
        
    # Use the package commands to run the job 
    if run_as_user=='': 
        # Run the modeling tools script for a preliminary setup 
        script+= 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'+delimiter
    else: 
        script += 'sudo runuser -l '+run_as_user+" -c 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'"+delimiter

    command = 'python /home/ec2-user/efs/Student-Voices/student_voices/run_coherence.py -c '+str(config)+' -cp '+str(config_path)+' -md '+str(model_dir)+' -s '+str(setting)+' -rd '+str(results_path)+' -nt '+str(ntop)+' -cg '+str(corpus_group)
        
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
    

def get_ldavis_script(config, setting, ntop, corpus_group, model_dir, config_path, fullpath, numwords, graphpath, desdir, vecdir, log_file_name, region, cancel_fleet=True, script='', run_as_user='', delimiter='\n'):
    # To be run within an instance 
    
    if script=='':    
        # Bash scripts submitted as user data must have the correct header 
        script = bash_scripts.init_userdata_script()
    
    script+= 'cd /home/ec2-user/efs/results'+delimiter 
        
    # Use the package commands to run the job 
    if run_as_user=='': 
        # Run the modeling tools script for a preliminary setup 
        script+= 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'+delimiter
    else: 
        script += 'sudo runuser -l '+run_as_user+" -c 'python /home/ec2-user/efs/Student-Voices/student_voices/modeling_tools.py'"+delimiter

    command = 'python /home/ec2-user/efs/Student-Voices/student_voices/run_lda_visualization.py -c '+str(config)+' -s '+str(setting)+' -cp '+str(config_path)+' -md '+str(model_dir)+' -nt '+str(ntop)+' -cg '+str(corpus_group)+' -fp '+str(fullpath)+' -nw '+str(numwords)+' -gp '+str(graphpath)+' -dd '+str(desdir)+' -vd '+str(vecdir)
        
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
