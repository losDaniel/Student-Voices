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
python lda_analysis.py

# # Pull the arguments to pass to the command and save the log 
# A1=$1
# SET=$2
# TXTFILE="LDA_$CON_$SET.txt"
# nohup sudo python lda_modeling.py -c $CON -s $SET &> $TXTFILE &
# curpid=$!

nohup python lda_modeling.py -c A1 -s LDA1 &> lda_A1_LDA1.txt &
curpid=$! 

# Wait until the previous job is done and then shutdown the instance 
nohup sh -c 'while ps -p $0 &> /dev/null; do sleep 10 ; done && sudo shutdown -h now ' $curpid &> run.txt
exit