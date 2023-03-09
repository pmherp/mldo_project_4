import subprocess
import os
import json

def apicalls():
    #Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1/"

    with open('config.json','r') as f:
        config = json.load(f) 

    dataset_csv_path = config['output_model_path']



    #Call each API endpoint and store the responses
    response1 = subprocess.run(['curl', 'http://127.0.0.1:8000/prediction?path=testdata/testdata.csv'], capture_output=True).stdout
    response2 = subprocess.run(['curl', 'http://127.0.0.1:8000/scoring'], capture_output=True).stdout
    response3 = subprocess.run(['curl', 'http://127.0.0.1:8000/summarystats'], capture_output=True).stdout
    response4 = subprocess.run(['curl', 'http://127.0.0.1:8000/diagnostics'], capture_output=True).stdout

    #combine all API responses
    responses = []
    responses.extend([response1, response2, response3, response4])

    #write the responses to your workspace
    with open(os.path.join(os.getcwd(), dataset_csv_path, 'apireturns2.txt'), 'w') as f:
        f.write(str(responses))
