import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import apicalls

import os
import json
import ast
import pickle
import pandas as pd


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
source_data = config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']


def run():
    ##################Check and read new data
    #first, read ingestedfiles.txt
    with open(os.path.join(os.getcwd(), dataset_csv_path, 'ingestedfiles.txt'), 'r') as f:
        ingested_files = ast.literal_eval(f.read())

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = set(os.listdir(os.path.join(os.getcwd(), source_data)))

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here

    if len(source_files.difference(ingested_files)) == 0:
        print('No new files found. Process ends.')
        return None

    ingestion.merge_multiple_dataframe()

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt'), 'r') as f:
        latest_score = ast.literal_eval(f.read())

    predicted, y = diagnostics.model_predictions(os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv'))

    #f1 = f1_score(predicted, y)
    f1 = scoring.score_model()
    #f1 = 0.112

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if(f1 >= latest_score):
        print('No retraining & redeployment needed: F1 score is higher or equal than latest score. Process ends.')
        return None

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    training.train_model()

    deployment.store_model_into_pickle()

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    reporting.score_model(predicted, y)

    apicalls.apicalls()


if __name__ == '__main__':
       run()