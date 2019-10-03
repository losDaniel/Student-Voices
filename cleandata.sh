# To be run within an instance 
# Run the data cleaning: 
# -- create the directories you need 
# -- pull scripts directory from github 
# -- run data cleaning 

cd efs 
cd scripts 
cd scripts 

# Run the modeling tools script for a preliminary setup 
python modeling_tools.py

# Pull the arguments to pass to the command and save the log 
# ARG=$1
# TXTFILE="clean_$ARG.txt"
# nohup python cleaning.py -c $ARG -o True &> $TXTFILE &

# Rigged to run with notebook
nohup python cleaning.py -c DC_CONFIG -o True &> DC_OUTFILE &
curpid=$!

# Wait until the previous job is done and then shutdown the instance 
nohup sh -c 'while ps -p $0 > /dev/null; do sleep 10; done && sudo shutdown -h now' $curpid