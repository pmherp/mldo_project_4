
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
test_data_path = config['test_data_path']
prod_deployment_path = config['prod_deployment_path']
output_folder_path = config['output_folder_path']

##################Function to get model predictions
def model_predictions(data):
    #read the deployed model and a test dataset, calculate predictions

    df = pd.read_csv(data)

    with open(os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    X = df.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1,3)
    y = df['exited'].values.reshape(-1,1).ravel()
    
    predicted = model.predict(X)

    return predicted, y

##################Function to get summary statistics
def dataframe_summary(pth):
    #calculate summary statistics here
    statistics_list = []
    df = pd.read_csv(pth)

    try:
        df = df.drop('Unnamed: 0', axis=1)
    except:
        pass

    numeric_cols = df.select_dtypes(include=['int64']).columns.tolist()

    for col in numeric_cols:
        mini = min(df[col])
        maxi = max(df[col])
        mean = np.mean(df[col])
        std = np.std(df[col])
        statistics_list.extend([mini, maxi, mean, std])

    return statistics_list

def missing_data(pth):
    nan_list = []
    df = pd.read_csv(pth)

    try:
        df = df.drop('Unnamed: 0', axis=1)
    except:
        pass

    nan = list(df.isna().sum())

    nan_comparison = [nan[i] / df.shape[0] for i in range(len(nan))]

    return nan_comparison


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    timer_list = []

    start_time_ingestion = timeit.default_timer()
    os.system('python ingestion.py')
    timing_ingestion = timeit.default_timer() - start_time_ingestion
    timer_list.append(timing_ingestion)

    start_time_training = timeit.default_timer()
    os.system('python training.py')
    timing_training = timeit.default_timer() - start_time_training
    timer_list.append(timing_training)

    return timer_list

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    output = subprocess.check_output(['pip', 'list', '--outdated'])

    lines = output.decode().split('\n')

    return lines


if __name__ == '__main__':
    predicted, _ = model_predictions(os.path.join(os.getcwd(), 'sourcedata', 'dataset3.csv'))
    print(f'Predictions: {predicted}\n')

    statistics_list = dataframe_summary(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'))
    print(f'Statistics List: {statistics_list}\n')

    nan_list = missing_data(os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv'))
    print(f'Percent Nan Values per Column: {nan_list}')

    timer_list = execution_time()
    print(f'Timer List: {timer_list}\n')

    lines = outdated_packages_list()
    for line in lines[2:-1]:
        package, current_version, latest_version, _ = line.split()
        print(f'{package}: {current_version} -> {latest_version}')
