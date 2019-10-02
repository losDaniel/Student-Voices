# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:56:39 2019

@author: carlo
"""
import spot_toolbox as spt
import sys, time, os

class spotted: 
    
    profiles={
            "default": {'firewall_ingress': ('tcp', 22, 22, '0.0.0.0/0'),      # Must define a firewall ingress rule in order to connect to an instance 
                        'image_id':'ami-0859ec69e0492368e',                    # Image ID from AWS. go to the launch-wizard to get the image IDs or use the boto3 client.describe_images() with Owners of Filters parameters to reduce wait time and find what you need.
                        'instance_type':'t2.micro',                            # Get a list of instance types and prices at https://aws.amazon.com/ec2/spot/pricing/ 
                        'price':'0.004',
                        'region':'us-west-2',                                  # All settings for us-west-2. Parameters (including prices) can change by region, make sure to review all parameters if changing regions.  
                        'scripts':[],                            
                        'username':'ec2-user',                                 # This will usually depend on the operating system of the image used. For a list of operating systems and defaul usernames check https://alestic.com/2014/01/ec2-ssh-username/
                        'efs_mount':True                                       # If true will check for an EFS mount in the instance and if one is not found it will create a file system or use an existing one and mount it. 
                        },
            "datasync":{'firewall_ingress': ('tcp', 22, 22, '0.0.0.0/0'),      # must enable nfs, http and other port ingress depending on endpoints https://docs.aws.amazon.com/datasync/latest/userguide/requirements.html#datasync-network
                        'image_id':'ami-0f2e06a04ee62ab37',                    # Datasync Image ID from AWS. List by region at https://docs.aws.amazon.com/datasync/latest/userguide/deploy-agents.html#ec2-deploy-agent 
                        'instance_type':'m5.2xlarge',                          # For recommended instance types for datasync https://docs.aws.amazon.com/datasync/latest/userguide/requirements.html#ec2-instance-types
                        'price': '0.15',                                       # Spot instance pricing list at https://aws.amazon.com/ec2/spot/pricing/ 
                        'region':'us-west-2',                                  # All settings for us-west-2. Datasync images vary by region, review all parameters if using a different region
                        'scripts':[],
                        'username':'ec2-user',
                        'efs_mount':False                                      # No need to mount an EFS on a datasync agent (ec2 Instance with a datasync image)
                        },
            "cleaning1":{'firewall_ingress': ('tcp', 22, 22, '0.0.0.0/0'),     
                        'image_id':'ami-0859ec69e0492368e',                    
                        'instance_type':'c5.4xlarge',                          # VCPU Compute Optimized Instance for data cleaning, bigger chip/more memory -> faster cleaning                     
                        'price':'0.30',
                        'region':'us-west-2',                                  
                        'scripts':[],                            
                        'username':'ec2-user',                                 
                        'efs_mount':True                                     
                        },
            "ldamodel1":{'firewall_ingress': ('tcp', 22, 22, '0.0.0.0/0'),     
                        'image_id':'ami-0859ec69e0492368e',                    
                        'instance_type':'c5d.9xlarge',                          # VCPU Compute Optimized Instance for data cleaning, bigger chip/more memory -> faster cleaning                     
                        'price':'0.60',
                        'region':'us-west-2',                                  
                        'scripts':[],                            
                        'username':'ec2-user',                                 
                        'efs_mount':True                                     
                        },
    }     

    name=''
    profile={}
    monitoring=True
    filesystem=''
    efs_mount=True
    newmount=''
    firewall=()
    image_id=''
    price=''
    region=''
    username=''
    key_pair=''
    sec_group=''    
    
    def __init__(self,
                 name,
                 profile='default',
                 monitoring=True,
                 filesystem='',
                 efs_mount=True,
                 newmount='',
                 firewall=(),
                 image_id='',
                 instance_type='',
                 price='',
                 region='',
                 username='',
                 key_pair='',
                 sec_group=''):
        '''
        A class to run, control and interact with spot instances. 
        __________
        parameters
        - name : string. name of the spot instance
        - profile : dict of settings for the spot instance
        - monitoring : bool, default True. set monitoring to True for the instance 
        - filesystem : string, default <name>. creation token for the EFS you want to connect to the instance  
        - image_id : Image ID from AWS. go to the launch-wizard to get the image IDs or use the boto3 client.describe_images() with Owners of Filters parameters to reduce wait time and find what you need.
        - instance_type : Get a list of instance types and prices at https://aws.amazon.com/ec2/spot/pricing/ 
        - price : float. maximum price willing to pay for the instance. 
        - region : string. AWS region
        - username : string. This will usually depend on the operating system of the image used. For a list of operating systems and defaul usernames check https://alestic.com/2014/01/ec2-ssh-username/
        - key_pair : string. name of the keypair to use. Will search for `key_pair`.pem in the current directory 
        - sec_group : string. name of the security group to use
        '''

        self.name=name 
        self.profile=spotted.profiles[profile] 
        self.monitoring=monitoring
        self.filesystem=filesystem
        self.newmount=newmount        
        
        if not efs_mount: 
            self.profile['efs_mount']=efs_mount
        if firewall!=():
            self.profile['firewall']=firewall
        if image_id!='':
            self.profile['image_id']=image_id
        if instance_type!='':
            self.profile['instance_type']=instance_type
        if str(price)!='':
            self.profile['price']=price
        if region!='':
            self.profile['region']=region
        if username!='':
            self.profile['username']=username
        if key_pair!='': 
            self.profile['key_pair']=profile
        if sec_group!='':
            self.profile['security_group']=sec_group                           
               
        print('', flush=True)
        print('#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#', flush=True)
        print('#~#~#~#~#~#~#~# Spot Instance: '+self.name, flush=True)
        print('#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#', flush=True)
        print('', flush=True)

        print(self.profile)
        try:                                     # Launch or connect to the spot instance under the given name
            # Returns the profile with any parameters that needed to be added automatically in order to connect (Key Pair and Security Group)                                                                 
            self.instance, self.profile = spt.launch_spot_instance(self.name, self.profile, self.monitoring)   
        except Exception as e:
            raise e
            sys.exit(1)
        
        if self.profile['efs_mount']: 
            print('Profile requesting EFS mount...')
            if self.filesystem=='':             # If no filesystem name is submitted 
                fs_name = self.name             # Retrieve or create a filesystem with the same name as the instance 
            else: 
                fs_name = self.filesystem     
            
            # Create and/or mount an EFS to the instance 
            try:                                
                self.mount_target, self.instance_dns, self.filesystem_dns = spt.retrieve_efs_mount(fs_name, self.instance, new_mount=self.newmount)
            except Exception as e: 
                raise e 
                sys.exit(1)        
                
            print('Connecting to instance to link EFS...')
            spt.run_script(self.instance, self.profile['username'], 'efs_mount.sh')
            
        if len(self.profile['scripts'])>0:
            spt.run_script(self.instance, self.profile['username'], self.profile['scripts'])
    
    def upload(self, files, remotepath):
        '''
        Upload a file or list of files to the instance. If an EFS is connected to the instance files can be uploaded to the EFS through the instnace. 
        __________
        parameters
        - files : str or list of str. file or list of files to upload
        - remotepath : str. path to upload files to, only one path can be specified. 
        '''
        if type(files)==str:
            files=[files]
        elif type(files)!=list: 
            raise TypeError('Files must but type str or list')

        st = time.time() 
            
        files_to_upload = [] 
        for file in files:
            files_to_upload.append(os.path.abspath(file))
        spt.upload_to_ec2(self.instance, self.profile['username'], files_to_upload, remote_dir=remotepath)    
    
        print('Time to Upload: %s' % str(time.time()-st))

    def run(self, scripts, cmd=False):
        '''
        Run a script or list of scripts
        __________
        parameters
        - scripts : str or list of strings. list of scripts files to run 
        - cmd : if True, each script in scripts is treated as an individual command
        '''
        st = time.time() 
        
        if type(scripts)==str:
            scripts=[scripts]
        elif type(scripts)!=list:
            raise TypeError('scripts must be string or list of strings')
        
        for script in scripts:
            if not cmd:
                print('\nExecuting script "%s"...' % str(script))
            try:
                if not spt.run_script(self.instance, self.profile['username'], script, cmd=cmd):
                    break
            except Exception as e: 
                print(str(e))
                print('Script %s failed with above error' % script)
    
        print('Time to Run Scripts: %s' % str(time.time()-st))

    def open_shell(self, port=22):
        '''Open an active shell'''
        spt.active_shell(self.instance, self.profile['username'])
    
    def terminate(self): 
        '''Terminate the instance'''
        spt.terminate_instance(self.instance['InstanceId'])                    