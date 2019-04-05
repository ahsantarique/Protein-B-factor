import sys
import numpy as np
import random
import os

if(len(sys.argv) != 2):
	print("usage: create_dataset window_size")
	exit(1)

random.seed(42)

MAX_FILE_COUNT = int(1e4)
SAMPLE_INTERVAL = 10

ws = int(sys.argv[1])
FEATURE_DIMENSION = ws*2+1
input_path_x = '../data/x'
input_path_y = '../data/y'

transform_y = lambda y: (y-np.min(y))/(np.max(y)-np.min(y)) 

def prepareDataSet(files):
	X = np.empty((0,FEATURE_DIMENSION), dtype='int8')
	y = np.empty((0,1), dtype='float')
	file_count = 0
	for file in files:
		file_count += 1
		if(file_count % SAMPLE_INTERVAL == 0):
			print('processing {}: {}'.format(file_count, file))
		if(file_count >= MAX_FILE_COUNT):
			break

		seq = np.append(np.append(np.zeros((ws,), dtype='int8') , np.load(os.path.join(input_path_x, file))) , np.zeros((ws,), dtype='int8'))
		bval = transform_y(np.load(os.path.join(input_path_y, file)))
	
		for i in range(len(seq) - FEATURE_DIMENSION+1):
			window = np.array([seq[i: i+FEATURE_DIMENSION]])
			#print(window.shape)
			X = np.append(X, window, axis = 0)
		
		y = np.append(y, np.array([bval]))
			
	print("x shape", X.shape, "y shape", y.shape )
	print(X[len(X)-1])
	print(X[0])
	return X , y
		
		

if __name__ == "__main__":
	all_files = os.listdir(input_path_x)
	random.shuffle(all_files)
	train_files = all_files[:int(0.8*len(all_files))]
	test_files = all_files[int(0.8*len(all_files)):]

	Xtrain, ytrain = prepareDataSet(train_files)
	Xtest, ytest = prepareDataSet(test_files)

	np.savez_compressed('Xtrain'+ str(ws) +'.npz', X=Xtrain)
	np.savez_compressed('Xtest'+ str(ws) +'.npz', X=Xtest)
	np.savez_compressed('ytrain'+ str(ws) +'.npz', y=ytrain)
	np.savez_compressed('ytest'+ str(ws) +'.npz', y=ytest)
	
