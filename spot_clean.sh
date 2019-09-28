# Run from the Home directory
# Launch an instance  
python spot_connect.py -n dta_clean_A1 -p cleaning1 -f reviewdata -s setup.sh
python spot_connect.py -n dta_clean_A1 -p cleaning1 -f reviewdata -s cleandata_A1.sh -t True
#python spot_connect.py -n cleanB1 -p cleaning1 -f reviewdata -s setup.sh
#python spot_connect.py -n cleanB1 -p cleaning1 -f reviewdata -s cleandata_B1.sh
python spot_connect.py -n dta_clean_C1 -p cleaning1 -f reviewdata -s setup.sh
python spot_connect.py -n dta_clean_C1 -p cleaning1 -f reviewdata -s cleandata_C1.shomm -t True 