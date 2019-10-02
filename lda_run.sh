#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
# Launch Instances to clean the data 
#~#~#~#~#~#~#~#~#~#~

# Launch an instance that connects to the reviewdata EFS drive and runs the cleaning scripts:
sh spot_lda.sh A1 LDA1

# Cleaning Instance for A1 Config
python spot_connect.py -n lda1 -p cleaning1 -f reviewdata -s scripts/instancesetup.sh
python spot_connect.py -n lda1 -p cleaning1 -f reviewdata -s scripts/genlda_A1.sh