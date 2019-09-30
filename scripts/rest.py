import time
import argparse

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Waiting')
	parser.add_argument('-t','--time',help='seconds to rest', default=20)

	args = parser.parse_args()
    
    os.mkdir(os.getcwd()+'/test/')

	time.sleep(int(args.time))