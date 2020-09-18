# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 07:38:09 2020

@author: puni3lv

Description:
    Train our model
    Create Web App using Flask
    Commit the code in Github
    Create an Account in Heroku(PAAS)
    Link the Github to Heroku
    Deploy the model
    Web App is ready
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('C:/Users/puni3lv/Downloads/Akash/Personal_Pr_Python/hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('C:/Users/puni3lv/Downloads/Akash/Personal_Pr_Python/model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('C:/Users/puni3lv/Downloads/Akash/Personal_Pr_Python/model.pkl','rb'))
print(model.predict([[2, 9, 6]]))