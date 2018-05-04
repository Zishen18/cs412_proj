import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

def cleanData(filename):
	data = pd.read_csv(filename, index_col=None)
	print("Converting numerical data into numerical data...")
	# convert Smoking string data to numerical numbers 
	for row in data.itertuples():
    		if data.loc[row.Index,'Smoking'] == 'never smoked':
        		data.loc[row.Index,'Smoking'] = 1.0
    		elif data.loc[row.Index,'Smoking'] == 'tried smoking':
        		data.loc[row.Index,'Smoking'] = 2.0
    		elif data.loc[row.Index,'Smoking'] == 'former smoker':
        		data.loc[row.Index,'Smoking'] = 3.0
    		elif data.loc[row.Index,'Smoking'] == 'current smoker':
        		data.loc[row.Index,'Smoking'] = 4.0

	# convert Alcohol string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Alcohol'] == 'never':
        		data.loc[row.Index,'Alcohol'] = 1.0
    		elif data.loc[row.Index,'Alcohol'] == 'social drinker':
        		data.loc[row.Index,'Alcohol'] = 2.0
    		elif data.loc[row.Index,'Alcohol'] == 'drink a lot':
        		data.loc[row.Index,'Alcohol'] = 3.0

	# convert Punctuality string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Punctuality'] == 'i am often running late':
        		data.loc[row.Index,'Punctuality'] = 1.0
    		elif data.loc[row.Index,'Punctuality'] == 'i am always on time':
        		data.loc[row.Index,'Punctuality'] = 2.0
    		elif data.loc[row.Index,'Punctuality'] == 'i am often early':
        		data.loc[row.Index,'Punctuality'] = 3.0

	# convert Lying string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Lying'] == 'never':
        		data.loc[row.Index,'Lying'] = 1.0
    		elif data.loc[row.Index,'Lying'] == 'only to avoid hurting someone':
        		data.loc[row.Index,'Lying'] = 2.0
    		elif data.loc[row.Index,'Lying'] == 'sometimes':
        		data.loc[row.Index,'Lying'] = 3.0
    		elif data.loc[row.Index,'Lying'] == 'everytime it suits me':
        		data.loc[row.Index,'Lying'] = 4.0


	# convert Internet usage string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Internet usage'] == 'no time at all':
        		data.loc[row.Index,'Internet usage'] = 1.0
    		elif data.loc[row.Index,'Internet usage'] == 'less than an hour a day':
        		data.loc[row.Index,'Internet usage'] = 2.0
    		elif data.loc[row.Index,'Internet usage'] == 'few hours a day':
        		data.loc[row.Index,'Internet usage'] = 3.0
    		elif data.loc[row.Index,'Internet usage'] == 'most of the day':
        		data.loc[row.Index,'Internet usage'] = 4.0

	# convert Gender string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Gender'] == 'female':
        		data.loc[row.Index,'Gender'] = 1.0
    		elif data.loc[row.Index,'Gender'] == 'male':
        		data.loc[row.Index,'Gender'] = 2.0

	# convert Left-right handed string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Left - right handed'] == 'right handed':
        		data.loc[row.Index,'Left - right handed'] = 1.0
    		elif data.loc[row.Index,'Left - right handed'] == 'left handed':
        		data.loc[row.Index,'Left - right handed'] = 2.0

	# convert Education string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Education'] == 'currently a primary school pupil':
        		data.loc[row.Index,'Education'] = 1.0
    		elif data.loc[row.Index,'Education'] == 'primary school':
        		data.loc[row.Index,'Education'] = 2.0
    		elif data.loc[row.Index,'Education'] == 'secondary school':
        		data.loc[row.Index,'Education'] = 3.0
    		elif data.loc[row.Index,'Education'] == 'college/bachelor degree':
        		data.loc[row.Index,'Education'] = 4.0
    		elif data.loc[row.Index,'Education'] == 'masters degree':
        		data.loc[row.Index,'Education'] = 5.0
    		elif data.loc[row.Index,'Education'] == 'doctorate degree':
        		data.loc[row.Index,'Education'] = 6.0

	# convert Only child string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Only child'] == 'yes':
        		data.loc[row.Index,'Only child'] = 1.0
    		elif data.loc[row.Index,'Only child'] == 'no':
        		data.loc[row.Index,'Only child'] = 2.0 


	# convert Village - town string data to numerical numbers
	for row in data.itertuples():
    		if data.loc[row.Index,'Village - town'] == 'village':
        		data.loc[row.Index,'Village - town'] = 1.0
    		elif data.loc[row.Index,'Village - town'] == 'city':
        		data.loc[row.Index,'Village - town'] = 2.0

	# convert House - block of flats string data to numerical numbers
	for row in data.itertuples():
		if data.loc[row.Index,'House - block of flats'] == 'block of flats':
			data.loc[row.Index,'House - block of flats'] = 1.0
		elif data.loc[row.Index,'House - block of flats'] == 'house/bungalow':
        		data.loc[row.Index,'House - block of flats'] = 2.0
	
	# fill missing data with most frequent value
	print("Filling missing data...")
	data = data.replace("nan", np.nan)
	data = data.replace("NaN", np.nan)
	imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
	imp.fit(data)
	data_trans = imp.transform(data)
	data = pd.DataFrame(data=data_trans[:,:], index=[i for i in range(len(data_trans))], 
                    columns=data.columns.tolist())

	# write to .csv file
	data.to_csv("clean_data.csv", index=False)

