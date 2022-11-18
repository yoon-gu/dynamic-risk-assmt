import json
import os
import sys
import training
import scoring
import ingestion
import deployment
import diagnostics
import reporting

##################Check and read new data
#first, read ingestedfiles.txt
with open('config.json','r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
ingested_file_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
with open(ingested_file_path, 'r') as ingested_file:
    ingested = ingested_file.readlines()
    ingested = [x.strip() for x in ingested]

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_dir = os.path.join(os.getcwd(), config['input_folder_path'])
input_files = os.listdir(input_dir)

all_input_files = list(set(ingested + input_files))
has_not_ingested_files = False;
if len(all_input_files) > len(ingested):
    has_not_ingested_files = True


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not has_not_ingested_files:
    print('there is no new ingested file')
    sys.exit()

new_data_frame = ingestion.merge_multiple_dataframe()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
prod_deployment_path = os.path.join(config['prod_deployment_path'])
latest_score_file = os.path.join(prod_deployment_path, 'latestscore.txt')
with open(latest_score_file, 'r') as f:
    msg = f.readline()
    prev_f1_score = float(msg)

new_f1_score = scoring.score_model(prod_deployment_path, new_data_frame)
if new_f1_score < prev_f1_score:
    model_drift = True
else:
    model_drift = False

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if not model_drift:
    sys.exit()

print(f"Model drift occurs. Prev f1 :{prev_f1_score}, New f1 {new_f1_score}")
training.train_model()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
os.system('python3 deployment.py')

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python3 diagnostics.py')
os.system('python3 reporting.py')
