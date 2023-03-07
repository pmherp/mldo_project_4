import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    directories = [input_folder_path]
    file_list = []

    final_df = pd.DataFrame(columns=['corporation', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited'])

    for directory in directories:
        file_names = os.listdir(os.getcwd()+directory)
        print(file_names)
        for each_file_name in file_names:
            file_list.append(each_file_name)
            current_df = pd.read_csv(os.getcwd()+directory+each_file_name)
            final_df = final_df.append(current_df).reset_index(drop=True)
    
    final_df = final_df.drop_duplicates()

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    final_df.to_csv(os.getcwd()+'/'+output_folder_path+'/finaldata.csv')

    with open(os.getcwd()+'/'+output_folder_path+'/ingestedfiles.txt', 'w') as f:
        f.write(str(file_list))


if __name__ == '__main__':
    merge_multiple_dataframe()
