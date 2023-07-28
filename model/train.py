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
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--update",type=str, default="False")
    parser.add_argument("--opt",type=str, default='auto')
    parser.add_argument("--batch",type=int, default=64)
    parser.add_argument("--mosaic",type=float, default=1.0)
    args = parser.parse_args()
    return args

config_path = '../model/config.yaml'
secret_path = '../model/secret.yaml'
config = load_config(config_path)
secret = load_config(secret_path)

current_version = config.get('dataset_version')
current_model_version = config.get('model_version')
args = parse_args()

# project 불러오기
rf = Roboflow(api_key=secret['api_key'])
project = rf.workspace(config['workspace']).project(config['project'])


if args.update=="True":
    current_version += 1
    current_model_version += 1
    config['dataset_version'] = current_version
    config['model_version'] = current_model_version
    upload.upload_data(config, project)
    generate_dataset.generate_dataset(project)

# dataset
dataset = roboflow_dataset(config['dataset_version'], project)
dataset = dataset.load_dataset()
config['data_dir']=dataset.location+'/data.yaml'
save_config(config, config_path)
#train
model = YOLO(config['model'])
results= model.train(data=config['data_dir'],epochs=args.epochs,patience=args.patience, optimizer=args.opt, batch = args.batch,mosaic=args.mosaic)
