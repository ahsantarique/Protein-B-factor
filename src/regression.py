import numpy as np
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import GradientBoostingRegressor as GBT
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from catboost import CatBoostRegressor as CBR
from sklearn.neural_network import MLPRegressor
import gc
import sys
import matplotlib.pyplot as plt

EXPONENT = 1/13
print("exp: ", EXPONENT)

def transform(y):
	return y
	#return (y-np.min(y))/(np.max(y)-np.min(y))
	#return (y-np.mean(y))/np.std(y)

def plot_true_and_prediction(y_true, y_pred):
	fig = plt.figure()
	plt.title('True Green, Predicted Yellow')
	plt.plot(y_pred, 'y-')
	plt.plot(y_true, 'g-')
	plt.show()


if(len(sys.argv) != 2):
	print('usage: regression.py WINDOW_SIZE')
	exit(1)

WINDOW_SIZE = int(sys.argv[1])
FEATURE_DIMENSION = WINDOW_SIZE* 2 + 1

Xtrain = np.load('Xtrain' + str(WINDOW_SIZE)+ '.npz')['X']
ytrain = np.load('ytrain' + str(WINDOW_SIZE)+ '.npz')['y']

ytrain = transform(ytrain**EXPONENT)
#print(ytrain[:200])
print("x shape", Xtrain.shape, "y shape", ytrain.shape )

# clf = LinearRegression()
clf = GBT(n_estimators = 2000, verbose = 4)
# clf = MLPRegressor()
#clf = RF(n_estimators=200, max_depth = 7, verbose = 5)
#clf = CBR(max_depth = 16, n_estimators = 200, verbose = 5)

print(clf)
clf.fit(Xtrain, ytrain)
ytrain_pred = clf.predict(Xtrain) ** 1/EXPONENT
print("Training PCC = ", np.corrcoef(ytrain**1/EXPONENT, ytrain_pred)[0, 1])
# clf.save_model('clf')

gc.collect()
Xtest = np.load('Xtest' + str(WINDOW_SIZE)+ '.npz')['X']
ytest = np.load('ytest' + str(WINDOW_SIZE)+ '.npz')['y']
ytest = transform(ytest)

ytest_pred = clf.predict(Xtest) ** (1/EXPONENT)

print("Testing PCC = ", np.corrcoef(ytest, ytest_pred)[0, 1])
plot_true_and_prediction(ytest[:200], ytest_pred[:200])
