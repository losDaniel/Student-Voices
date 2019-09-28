# Run from the Home directory

# Launch an instance  
python spot_connect.py -n cleanA1 -p cleaning1 -f reviewdata -s setup.sh
python spot_connect.py -n cleanA1 -p cleaning1 -f reviewdata -s cleandata_A1.sh

python spot_connect.py -n cleanB1 -p cleaning1 -f reviewdata -s setup.sh
python spot_connect.py -n cleanB1 -p cleaning1 -f reviewdata -s cleandata_B1.sh

python spot_connect.py -n cleanC1 -p cleaning1 -f reviewdata -s setup.sh
python spot_connect.py -n cleanC1 -p cleaning1 -f reviewdata -s cleandata_C1.sh