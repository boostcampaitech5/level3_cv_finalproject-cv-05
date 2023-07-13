from ultralytics import YOLO
from argparse import ArgumentParser
from load_dataset import roboflow_dataset
import yaml
import upload
import generate_dataset
from roboflow import Roboflow

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, file_path):
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs",type=int, default=3000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--update",type=bool, default=False)
    args = parser.parse_args()
    return args

config_path = 'config.yaml'
secret_path = 'secret.yaml'
config = load_config(config_path)
secret = load_config(secret_path)

current_version = config.get('dataset_version')
args = parse_args()

# project 불러오기
rf = Roboflow(api_key=secret['api_key'])
project = rf.workspace(config['workspace']).project(config['project'])


if args.update:
    current_version += 1
    config['dataset_version'] = current_version
    upload.upload_data(config, project)
    generate_dataset.generate_dataset(project)
    save_config(config, config_path)

# dataset
dataset = roboflow_dataset(config['dataset_version'], project)
dataset = dataset.load_dataset()

#train
model = YOLO(config['model'])
results= model.train(data=dataset.location+'/data.yaml',epochs=args.epochs,patience=args.patience)
