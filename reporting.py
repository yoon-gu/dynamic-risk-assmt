import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    testdatacsv = os.path.join(test_data_path, 'testdata.csv')
    test_df = pd.read_csv(testdatacsv)
    y_true = test_df['exited'].values
    y_pred = model_predictions(testdatacsv)
    cofusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    df_cfm = pd.DataFrame(cofusion_matrix, index = ["0", "1"], columns = ["0", "1"])
    cfm_plot = sns.heatmap(df_cfm, annot=True)
    cfm_path = os.path.join(model_path, 'confusionmatrix.png')
    cfm_plot.figure.savefig(cfm_path)


if __name__ == '__main__':
    score_model()
