from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
test_data_path = config['test_data_path']
output_model_path = config['output_model_path']


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    file_name = os.listdir(os.path.join(os.getcwd(),test_data_path))
    model_name = os.listdir(os.path.join(os.getcwd(), output_model_path))

    test_data = pd.read_csv(os.path.join(os.getcwd(), test_data_path, str(file_name[0])))

    with open(os.path.join(os.getcwd(), output_model_path, str(model_name[0])), 'rb') as f:
        model = pickle.load(f)
    
    X = test_data.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1,3)
    y = test_data['exited'].values.reshape(-1,1).ravel()
    
    predicted = model.predict(X)

    f1 = f1_score(predicted, y)
    
    with open(os.path.join(os.getcwd(), output_model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1))
