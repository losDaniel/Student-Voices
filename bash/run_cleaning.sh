#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
# Launch Instances to clean the data 
#~#~#~#~#~#~#~#~#~#~

# Launch an instance that connects to the reviewdata EFS drive and runs the cleaning scripts:
cd ..

# # Cleaning Instance for A1 Config
python spot_connect.py -n cleaning_A1 -p cleaning1 -f reviewdata -s bash/instancesetup.sh
python spot_connect.py -n cleaning_A1 -p cleaning1 -f reviewdata -s bash/script_pull.sh
ptyhon scripts/set_cleandata.py -c A1
python spot_connect.py -n cleaning_A1 -p cleaning1 -f reviewdata -s bash/cleantemp.sh

# # Cleaning Instance for B1 Config
python spot_connect.py -n cleaning_B1 -p cleaning1 -f reviewdata -s bash/instancesetup.sh
python spot_connect.py -n cleaning_B1 -p cleaning1 -f reviewdata -s bash/script_pull.sh
ptyhon scripts/set_cleandata.py -c B1
python spot_connect.py -n cleaning_B1 -p cleaning1 -f reviewdata -s bash/cleantemp.sh