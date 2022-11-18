import pickle
import subprocess
import pandas as pd
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])

##################Function to get model predictions
def model_predictions():
    #read the deployed model and a test dataset, calculate predictions
    prod_deployment_path = os.path.join(config['prod_deployment_path']) 

    model_file_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    testdata = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X = testdata[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = testdata['exited'].values.reshape(-1, 1)
    predicted = model.predict(X)

    return predicted

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    data_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(data_path)

    statistics = [
        np.median(df['lastmonth_activity']),
        np.mean(df['lastmonth_activity']),
        np.std(df['lastmonth_activity']),
        np.median(df['lastyear_activity']),
        np.mean(df['lastyear_activity']),
        np.std(df['lastyear_activity']),
        np.median(df['number_of_employees']),
        np.mean(df['number_of_employees']),
        np.std(df['number_of_employees']),
        ]
    return statistics

#################Function to investigate missing data
def investigation_missing_data():
    data_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(data_path)

    nas = list(df.isna().sum())
    na_ratio = [nas[i] / len(df.index) for i in range(len(nas))]
    return na_ratio

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    starttime = timeit.default_timer()
    os.system("python3 ingestion.py")
    ingestion_timing = timeit.default_timer() - starttime

    starttime = timeit.default_timer()
    os.system("python3 training.py")
    training_timing = timeit.default_timer() - starttime
    return ingestion_timing, training_timing

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    df = pd.DataFrame(columns=['package_name', 'current', 'recent'])

    with open("requirements.txt", "r") as file:
        strings = file.readlines()
        names = []
        cur = []
        recent = []

        for line in strings:
            name, cur_ver = line.strip().split('==')
            names.append(name)
            cur.append(cur_ver)
            info = subprocess.check_output(['python', '-m', 'pip', 'show', name])
            recent.append(str(info).split('\\n')[1].split()[1])

        df['package_name'] = names
        df['current'] = cur
        df['recent_available'] = recent


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    investigation_missing_data()
    execution_time()
    outdated_packages_list()
