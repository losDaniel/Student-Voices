import time
import argparse
import os 

# A function to modify the cleandata.py bash script to work with whatever argument we pass
def set_cleandata(config):
    
    with open('cleandata.sh', 'r') as f: 
        txt = f.read()
        txt = txt.replace('DC_CONFIG',config)
        txt = txt.replace('DC_OUTFILE','clean_'+config+'.txt')

    with open('cleantemp.sh', 'w') as w: 
        w.write(txt)
        w.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Waiting')
    parser.add_argument('-c','--config',help='seconds to rest', default=20)
    args = parser.parse_args()

    set_cleandata(args.config)