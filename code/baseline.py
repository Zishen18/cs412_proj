import data_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

data = pd.read_csv("new_data.csv", index_col=None)
Y = data["Empathy"]
X = data.drop(['Empathy'], axis=1)
print("X shape = ", X.shape)
X_tr, X_dev, X_te, Y_tr, Y_dev, Y_te = data_split.splitdata(X, Y, 0.6, 0.2)
Xtr = X_tr.as_matrix()
Ytr = Y_tr.as_matrix()
Xte = X_te.as_matrix()
Yte = Y_te.as_matrix()
N = Y_te.size
pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(Xtr, Ytr).predict(Xte)
error = np.sqrt((np.sum((pred - Yte)**2))/N)
print(error)
