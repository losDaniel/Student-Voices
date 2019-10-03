import time
import argparse
import os 

# A function to modify the cleandata.py bash script to work with whatever argument we pass
def set_ldaexec(config, setting):
    
    with open('bash/ldaexec.sh', 'r') as f: 
        txt = f.read()
        txt = txt.replace('DC_CONFIG',config)
        txt = txt.replace('SETTING',setting)
        txt = txt.replace('OUTFILE','lda_'+str(config)+'_'+str(setting)+'.txt')

    with open('bash/ldatemp.sh', 'w') as w: 
        w.write(txt)
        w.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Waiting')
    parser.add_argument('-c','--config',help='data configuration',required=True)
    parser.add_argument('-s','--setting',help='LDA settings to use',required=True)
    args = parser.parse_args()

    set_cleandata(args.config)