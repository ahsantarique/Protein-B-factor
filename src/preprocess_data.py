import sys
import os
import numpy as np

if(len(sys.argv) != 3):
	print("usage: python3 preprocess_data.py path/to/input" "path/to/output")
	exit(1)
amino_acids = dict(CYS = 0,
TRP = 1,
MET = 2,
HIS = 3,
GLN = 4,
TYR = 5,
PHE = 6,
ASN = 7,
PRO = 8,
ARG = 9,
ILE = 10,
THR = 11,
LYS = 12,
ASP = 13,
SER = 14,
GLU = 15,
VAL = 16,
GLY = 17,
ALA = 18,
LEU = 19,
other = 20
)

input_path = sys.argv[1]
output_path = sys.argv[2]

MAX_FILE_COUNT = int(5e2)
PROGRESS_PERIOD = 10

file_count = 0

for file in os.listdir(input_path):
	if(file_count == MAX_FILE_COUNT):
		break
	file_count += 1	
	if(file_count % PROGRESS_PERIOD == 0):
		print("processed: {} files".format(file_count))
	
	data = np.loadtxt(os.path.join(input_path, file), dtype='str')
	y = np.array(data[:,1], dtype = 'float16')
	
	if(any(y <= 0)):
		#y values must be positive
		continue
	#y = (y - y.mean()) / y.std()

	enumerate = lambda data: np.array([amino_acids[i] if i in amino_acids else 20 for i in data], dtype = 'int8')
	x = enumerate(data[:, 0])

	np.save(os.path.join(output_path, 'x', file), x)
	np.save(os.path.join(output_path, 'y', file), y)

	
