import pandas as pd
import numpy as np
import math

def dataEngineering(filename):
	data = pd.read_csv(filename, index_col=None)

	# create feature BMI
	print("creating feature BMI...")
	hwc = list(zip(data["Height"], data["Weight"]))
	bmi = []
	for h, w in hwc:
		if(str(h) == "nan" or str(w) == "nan"):
			bmi.append("nan")
		ret = float(w/math.pow(h/100, 2))
		bmi.append(ret)

	data["BMI"] = bmi

	# convert BMI to bins
	for row in data.itertuples():
    		if str(data.loc[row.Index,'BMI']) == 'nan':
        		continue
    		elif data.loc[row.Index,'BMI'] < 18.5:
        		data.loc[row.Index,'BMI'] = 1.0
    		elif data.loc[row.Index,'BMI'] < 25.0:
        		data.loc[row.Index,'BMI'] = 2.0
    		elif data.loc[row.Index,'BMI'] < 30.0:
        		data.loc[row.Index,'BMI'] = 3.0
    		elif data.loc[row.Index,'BMI'] < 35.0:
        		data.loc[row.Index,'BMI'] = 4.0
    		elif data.loc[row.Index,'BMI'] >= 35.0:
        		data.loc[row.Index,'BMI'] = 5.0

	# drop "Height" and "Weight" columns
	data = data.drop(['Height', 'Weight'], axis=1)

	# convert Age to bins
	print("Converting Age to bins...")
	for row in data.itertuples():
    		if str(data.loc[row.Index,'Age']) == 'nan':
        		continue
    		elif data.loc[row.Index,'Age'] <= 18:
        		data.loc[row.Index,'Age'] = 1.0
    		elif data.loc[row.Index,'Age'] <= 23.0:
        		data.loc[row.Index,'Age'] = 2.0
    		elif data.loc[row.Index,'Age'] <= 27.0:
        		data.loc[row.Index,'Age'] = 3.0
    		elif data.loc[row.Index,'Age'] <= 30.0:
        		data.loc[row.Index,'Age'] = 4.0


	# create feature music: asemble of original features realted with music
	# sum the preference of users for each kind of music to indicate their preference of music
	print("Creating feature music...")
	music = data[['Music', 'Slow songs or fast songs', 'Dance', 'Folk', 'Country', 
            'Classical music', 'Musical', 'Pop', 'Rock', 'Metal or Hardrock',
             'Punk', 'Hiphop, Rap', 'Reggae, Ska', 'Swing, Jazz', 'Rock n roll', 'Alternative',
             'Latino', 'Techno, Trance', 'Opera']]
	data['music'] = music.sum(axis=1)
	data = data.drop(['Music', 'Slow songs or fast songs', 'Dance', 'Folk', 'Country', 
            'Classical music', 'Musical', 'Pop', 'Rock', 'Metal or Hardrock',
             'Punk', 'Hiphop, Rap', 'Reggae, Ska', 'Swing, Jazz', 'Rock n roll', 'Alternative',
             'Latino', 'Techno, Trance', 'Opera'], axis=1)


	# create feature movie: asemble of original features realted with movie
	# sum the preference of users for each kind of movie to indicate their preference of movie
	print("Creating feature movie...")
	movie = data[['Movies', 'Horror', 'Thriller', 'Comedy', 'Romantic', 
            'Sci-fi', 'War', 'Fantasy/Fairy tales', 'Animated', 'Documentary',
             'Western', 'Action']]
	data['movie'] = movie.sum(axis=1)
	data = data.drop(['Movies', 'Horror', 'Thriller', 'Comedy', 'Romantic', 
            'Sci-fi', 'War', 'Fantasy/Fairy tales', 'Animated', 'Documentary',
             'Western', 'Action'], axis=1)

	# create feature fitness: asemble of original features realted with fitness
	# fitness = data['Healthy eating'] / (data['Smoking'] * data['Alcohol'])
	print("Creating feature fitness...")
	fitness = data[['Smoking', 'Alcohol', 'Healthy eating']]
	fit = []
	for row in data.itertuples():
    		if str(data.loc[row.Index,'Smoking']) == 'nan' or str(data.loc[row.Index,'Alcohol']) == 'nan' or str(data.loc[row.Index,'Healthy eating']) == 'nan':
        		fit.append("nan")
    		val = data.loc[row.Index,'Healthy eating'] / (data.loc[row.Index,'Smoking'] * 
                                                  data.loc[row.Index,'Alcohol'])
    		fit.append(val)
	data["fitness"] = fit
	data = data.drop(['Smoking', 'Alcohol', 'Healthy eating'], axis=1)

	# create feature expenditure: asemble of original features realted with expenditure
	# sum of spending for shopping centres, Entertainment, looks, gadgets
	print("Creating feature expenditure...")
	expenditure = data[['Shopping centres', 'Entertainment spending', 'Spending on looks',
               'Spending on gadgets', 'Spending on healthy eating']]
	data['expenditure'] = expenditure.sum(axis=1)
	data = data.drop(['Shopping centres', 'Entertainment spending', 'Spending on looks',
               'Spending on gadgets', 'Spending on healthy eating'], axis=1)

	data.to_csv("new_data.csv", index=False)

