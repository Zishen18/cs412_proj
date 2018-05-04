import data_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("new_data.csv", index_col=None)
Y = data["Empathy"]
X = data.drop(['Empathy'], axis=1)
X_tr, X_dev, X_te, Y_tr, Y_dev, Y_te = data_split.splitdata(X, Y, 0.6, 0.2)
rfc = RandomForestClassifier(n_estimators=2000)
rfc_model = rfc.fit(X_tr, Y_tr)
pred = rfc_model.predict(X_te)
Y_te_mat = Y_te.as_matrix()
print(pred)
print(Y_te_mat)
N = Y_te.size
error = np.sqrt((np.sum((pred - Y_te_mat)**2))/N)
print(error)
