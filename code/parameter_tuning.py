import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
import warnings
import data_split
warnings.filterwarnings("ignore")

def tune_parameter(X_tr, Y_tr, X_dev, Y_dev):
	
	learning_rate = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5]
	err = []
	lr_b = None
	err_m = float('inf')
	for lr in learning_rate:
		model = xgb.XGBClassifier(learning_rate =lr,n_estimators=1000, max_depth=150,
                        min_child_weight=50, gamma=0.2,subsample=0.8, colsample_bytree=0.4, objective= 'multi:softmax',
                        nthread=4, scale_pos_weight=50, seed=27)
		tr_model = model.fit(X_tr, Y_tr)
		pred = tr_model.predict(X_dev)
		Y_dev_mat = Y_dev.as_matrix()
		N = Y_dev.size
		error = np.sqrt((np.sum((pred - Y_dev_mat)**2))/N)
		err.append(error)
		if error < err_m:
			err_m = error
			lr_b = lr
	plt.plot(learning_rate, err, 'r-')
	plt.ylabel('Error')
	plt.xlabel('learning_rate')
	plt.title('Error VS learning_rate')
	print("learning_rate = ", lr_b)
        #print("err_m = ", err_m)
	plt.savefig("Error VS learning_rate")

	# tune parameter: n_estimator
	num_est = [10, 50, 80, 100, 200, 300, 400, 450, 500, 800, 1000, 1500]
	err = []
	est_b = None
	err_m = float('inf')
	for i in num_est:
		model = xgb.XGBClassifier(learning_rate =lr_b,n_estimators=i, max_depth=150,
			min_child_weight=50, gamma=0.2,subsample=0.8, colsample_bytree=0.4, objective= 'multi:softmax', 
			nthread=4, scale_pos_weight=50, seed=27)
		tr_model = model.fit(X_tr, Y_tr)
		pred = tr_model.predict(X_dev)
		Y_dev_mat = Y_dev.as_matrix()
		N = Y_dev.size
		error = np.sqrt((np.sum((pred - Y_dev_mat)**2))/N)
		err.append(error)
		if error < err_m:
			err_m = error
			est_b = i
	plt.plot(num_est, err, 'r-')
	plt.ylabel('Error')
	plt.xlabel('n_estimators')
	plt.title('Error VS n_estimators')
	print("n_estimators = ", est_b)
	#print("err_m = ", err_m)
	plt.savefig("Error VS n_estimators")
	
	# tune parameter: min_child_weight
	min_child_weight = [3, 5, 10, 30, 40, 50, 55, 60, 80, 100, 120, 150, 180]
	err = []
	mcw_b = None
	err_m = float('inf')
	for mcw in min_child_weight:
                model = xgb.XGBClassifier(learning_rate =lr_b,n_estimators=est_b, max_depth=150,
                        min_child_weight=mcw, gamma=0.2,subsample=0.8, colsample_bytree=0.4, objective= 'multi:softmax',
                        nthread=4, scale_pos_weight=50, seed=27)
                tr_model = model.fit(X_tr, Y_tr)
                pred = tr_model.predict(X_dev)
                Y_dev_mat = Y_dev.as_matrix()
                N = Y_dev.size
                error = np.sqrt((np.sum((pred - Y_dev_mat)**2))/N)
                err.append(error)
                if error < err_m:
                        err_m = error
                        mcw_b = mcw
	plt.plot(min_child_weight, err, 'r-')
	plt.ylabel('Error')
	plt.xlabel('min_child_weight')
	plt.title('Error VS min_child_weight')
	print("min_child_weight = ", mcw_b)
        #print("err_m = ", err_m)
	plt.savefig("Error VS min_child_weight")
	
	# tune parameter: min_child_weight
	gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	err = []
	gamma_b = None
	err_m = float('inf')
	for g in gamma:
		model = xgb.XGBClassifier(learning_rate =lr_b,n_estimators=est_b, max_depth=150,
                        min_child_weight=mcw_b, gamma=g,subsample=0.8, colsample_bytree=0.4, objective= 'multi:softmax',
                        nthread=4, scale_pos_weight=50, seed=27)
		tr_model = model.fit(X_tr, Y_tr)
		pred = tr_model.predict(X_dev)
		Y_dev_mat = Y_dev.as_matrix()
		N = Y_dev.size
		error = np.sqrt((np.sum((pred - Y_dev_mat)**2))/N)
		err.append(error)
		if error < err_m:
			err_m = error
			gamma_b = g
	plt.plot(gamma, err, 'r-')
	plt.ylabel('Error')
	plt.xlabel('gamma')
	plt.title('Error VS gamma')
	print("gamma = ", gamma_b)
        #print("err_m = ", err_m)
	plt.savefig("Error VS gamma")
	
	# tune parameter: scale_pos_weight
	colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	err = []
	cbt_b = None
	err_m = float('inf')
	for cbt in colsample_bytree:
		model = xgb.XGBClassifier(learning_rate =lr_b,n_estimators=est_b, max_depth=150,
                        min_child_weight=mcw_b, gamma=gamma_b,subsample=0.8, colsample_bytree=cbt, objective= 'multi:softmax',
                        nthread=4, scale_pos_weight=50, seed=27)
		tr_model = model.fit(X_tr, Y_tr)
		pred = tr_model.predict(X_dev)
		Y_dev_mat = Y_dev.as_matrix()
		N = Y_dev.size
		error = np.sqrt((np.sum((pred - Y_dev_mat)**2))/N)
		err.append(error)
		if error < err_m:
			err_m = error
			cbt_b = cbt
	plt.plot(colsample_bytree, err, 'r-')
	plt.ylabel('Error')
	plt.xlabel('colsample_bytree')
	plt.title('Error VS colsample_bytree')
	print("colsample_bytree = ", cbt_b)
        #print("err_m = ", err_m)
	plt.savefig("Error VS colsample_bytree")
	
	return lr_b, est_b, mcw_b, gamma_b, cbt_b

