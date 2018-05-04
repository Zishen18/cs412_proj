import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

def splitdata(X, Y, trsize, tesize):
	X, X_te, Y, Y_te = train_test_split(X, Y, test_size=tesize, train_size=1 - tesize)
	tsize = trsize / (1 - tesize)
	dsize = 1 - tsize
	X_tr, X_dev, Y_tr, Y_dev = train_test_split(X, Y, test_size=dsize, train_size=tsize)
	return X_tr, X_dev, X_te, Y_tr, Y_dev, Y_te

#data = pd.read_csv("new_data.csv", index_col=None)
#Y = data["Empathy"]
#X = data.drop(['Empathy'], axis=1)
#print("X shape = ", X.shape)
#X_tr, X_dev, X_te, Y_tr, Y_dev, Y_te = splitdata(X, Y, 0.6, 0.2)

#print("data shape = ", data.shape)
#print("X_tr shape = ", X_tr.shape)
#print("Y_tr shape = ", Y_tr.shape)
#print("X_te shape = ", X_te.shape)
#print("Y_te shape = ", Y_te.shape)
#print("X_dev shape = ", X_dev.shape)
#print("Y_dev shape = ", Y_dev.shape)



