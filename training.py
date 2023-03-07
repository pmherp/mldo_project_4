from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
model_path = config['output_model_path']


#################Function for training the model
def train_model():
    file_name = os.listdir(os.path.join(os.getcwd(),dataset_csv_path))
    training_data = pd.read_csv(os.path.join(os.getcwd(),dataset_csv_path,str(file_name[0])))
    training_data = training_data.drop('corporation', axis=1)

    X = training_data.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1,3)
    y = training_data['exited'].values.reshape(-1,1).ravel()
    
    #use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model = logit.fit(X, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    pickle.dump(model, open(os.getcwd()+'/'+model_path+'/trainedmodel.pkl', 'wb'))

if __name__ == '__main__':
    train_model()