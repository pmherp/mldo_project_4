from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import ast



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
output_model_path = config['output_model_path']


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    file_name = os.listdir(os.path.join(os.getcwd(),dataset_csv_path))
    model_name = os.listdir(os.path.join(os.getcwd(), output_model_path))
    print(model_name)

    with open(os.path.join(os.getcwd(), output_model_path, str(model_name[0])), 'rb') as f:
        model = pickle.load(f)
        
    with open(os.path.join(os.getcwd(), output_model_path, str(model_name[1])), 'r') as f:
        latest_score = ast.literal_eval(f.read()) 

    with open(os.path.join(os.getcwd(), dataset_csv_path, str(file_name[1])), 'r') as f:
        ingested_files = ast.literal_eval(f.read())
    
    if not os.path.isdir(prod_deployment_path):
        os.mkdir(prod_deployment_path)
    
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'w') as f:
        f.write(str(model))
    
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt'), 'w') as f:
        f.write(str(latest_score))
    
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'ingestedfiles.txt'), 'w') as f:
        f.write(str(ingested_files))
