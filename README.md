# mldo_project_4: Dynamic Risk Assessment

This repo is created to build a completed Machine Learning pipeline from Data Ingestion, Training, scoring, and deploying, plus Process Automation

## Dependencies

All dependencies is listed in the requirements.txt file.

## Data Ingestion

Automatically detect all the csv files in the directory specified in the input folder path - used for training. Then, combining all these in a single dataframe and de-dupe the dataframe to ensure that it contains unique rows.

## Training, Scoring, Deploying

1. Model training: Train an ML model that predicts attrition risk.
2. Model Scoring: Scoring model performance by calculating F1 score of the trained model on a testing data.
3. Model deploying: Normally copy the trained model (pickle file), model score (txt file) and a recored data of the ingested data to a production deployment direction.

## Model Diagnostics

1. Missing data: Detect missing data (NA values), count the number of NA values in each column in the dataset and calculate what percent of each column consists of NA values.
2. Timing: Time measurement for data ingestion, and model training.
3. Dependencies: Check whether the module dependencies are up-to-date, output results as a table with three columns: the first column will show the name of a Python module that you're using; the second column will show the currently installed version of that Python module, and the third column will show the most recent available version of that Python module.

## Reporting

1. Generating Plots: Plot confusion matrix on obtained predicted values and actual values for the data.
2. Flask API setup: Create API to easily access ML diagnostics and results with four endpoints: one for model predictions, one for model scoring, one for summary statistics, and one for other diagnostics.

## Automation

1. Checking and Reading New Data: check whether any new data exists that need to be ingested (new file (if exists) appears in input_folder_path)
2. Deciding whether to proceed (first time): If previous step is no new data, there will be no need to check for model drift, if not, need to continue to next step.
3. Checking for Model Drift: Evaluating model performance on new data, and comparing with model prediction on previous data (based on recored model score). If the score id lower, compares to precious one, the model drift has occured. Otherwise, it has not.
4. Deciding whether to proceed (second time): If in the step 3, there is no model drift, it means that the current model is working well. Otherwise, we need to move to next step.
5. Re-training: Train a new model using the most recent data which is obtained from the previous "Checking and Reading new data" step. Using training.py to complete this step, and a model trained on the most recent data will be saved in the workspace.
6. Re-deployment: Run script deployment.py to deploy the new trained model.
7. Diagnostics and Reporting: Use script reporting.py and apicalls.py on the most recently deployed model.

## Process Automation

Cron job for the Full Pipeline: Write a crontab file that runs the fullprocess.py script one time every, say, X min.
