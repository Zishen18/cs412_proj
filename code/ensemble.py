import data_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import warnings
import data_clean
import data_engineering
import parameter_tuning
warnings.filterwarnings("ignore")

# Clean original data
orig_data = "../data/responses.csv"
print("Step 1: Clean original data...")
data_clean.cleanData(orig_data)

# Data Engineering
clean_data = "clean_data.csv"
print("Step 2: Data Engineering ...")
data_engineering.dataEngineering(clean_data)

# Data Split
print("Step 3: Split data into training data, development data, test data")
data = pd.read_csv("new_data.csv", index_col=None)
Y = data["Empathy"]
X = data.drop(['Empathy'], axis=1)
print("X shape = ", X.shape)
X_tr, X_dev, X_te, Y_tr, Y_dev, Y_te = data_split.splitdata(X, Y, 0.6, 0.2)

# baseline
print("Step 4: Baseline...")
dt_model = DecisionTreeClassifier(max_depth = 15).fit(X_tr, Y_tr)
pred = dt_model.predict(X_te)
N = Y_te.size
baseline_error = np.sqrt((np.sum((pred - Y_te)**2))/N)
print("baseline RMSE = ", baseline_error)

# Hyperparameters tuning
print("Step 5: Tune parameters: use cross validation to tune parameters...")
lr, est, mcw, gamma, cbt = parameter_tuning.tune_parameter(X_tr, Y_tr, X_dev, Y_dev)

# modeling
print("Step 6: train XGBClassifier model...")
model = xgb.XGBClassifier(learning_rate =lr,
             n_estimators=est, max_depth=50,
             min_child_weight=mcw, gamma=gamma,
             subsample=0.8, colsample_bytree=cbt,
             objective= 'multi:softmax', nthread=4,
             scale_pos_weight=50, seed=27)
tr_model = model.fit(X_tr, Y_tr)

print("Step 7: using the trained model to make prediction on test data...")
pred = tr_model.predict(X_te)

print("Step 8: Evaluation. Compute the root mean squared error(RMSE)")
Y_te_mat = Y_te.as_matrix()
print("prediction = ", pred)
print("ground truth = ", Y_te_mat)
N = Y_te.size
print("size of N = ", N)
error = np.sqrt((np.sum((pred - Y_te_mat)**2))/N)
print("RMSE = ", error)

print("Step 9: Try our approach in dev data...")
pred_dev = tr_model.predict(X_dev)
Y_dev_mat = Y_dev.as_matrix()
print("development data prediction = ", pred_dev)
print("development ground truth = ", Y_dev_mat)

