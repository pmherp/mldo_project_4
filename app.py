from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    path = request.args.get('path')
    predicted, _ = model_predictions(os.path.join(os.getcwd(), str(path)))

    return str(predicted)


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats1():        
    #check the score of the deployed model
    f1 = score_model()
    return str(f1)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats2():        
    #check means, medians, and modes for each column
    statistics_list = dataframe_summary(os.path.join(os.getcwd(), 'testdata', 'testdata.csv'))
    return str(statistics_list)


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def stats3():        
    #check timing and percent NA values
    nan_list = missing_data(os.path.join(os.getcwd(), 'testdata', 'testdata.csv'))
    timer_list = execution_time()
    lines = outdated_packages_list()
    for line in lines[2:-1]:
        package, current_version, latest_version, _ = line.split()
        #print(f'{package}: {current_version} -> {latest_version}')

    return [str(nan_list), str(timer_list), [str(package), str(current_version), str(latest_version)]]


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
