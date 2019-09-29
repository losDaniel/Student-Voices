#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
# Launch Instances to clean the data 
#~#~#~#~#~#~#~#~#~#~

# Launch an instance that connects to the reviewdata EFS drive and runs the cleaning scripts:

# Cleaning Instance for A1 Config
python spot_connect.py -n cleaning_A1 -p cleaning1 -f reviewdata -s scripts/cleansetup.sh
python spot_connect.py -n cleaning_A1 -p cleaning1 -f reviewdata -s scripts/cleandata_A1.sh

# Cleaning Instance for B1 Config 
#python spot_connect.py -n dtac_B1 -p cleaning1 -f reviewdata -s scripts/cleansetup.sh
#python spot_connect.py -n dtac_B1 -p cleaning1 -f reviewdata -s scripts/cleandata_A1.sh

# Cleaning Instance for C1 Config 
#python spot_connect.py -n datac_C1 -p cleaning1 -f reviewdata -s scripts/cleansetup.sh
#python spot_connect.py -n datac_C1 -p cleaning1 -f reviewdata -s scripts/cleandata_C1.sh