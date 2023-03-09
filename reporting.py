import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_model_path']


##############Function for reporting
def score_model(predicted, y):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    cm = confusion_matrix(y, predicted)

    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if not os.path.isdir(dataset_csv_path):
        os.mkdir(dataset_csv_path)

    plt.savefig(os.path.join(os.getcwd(), dataset_csv_path, 'confusionmatrix2.png'))


if __name__ == '__main__':
    predicted, y = model_predictions(os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv'))
    score_model(predicted, y)
