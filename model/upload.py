from roboflow import Roboflow
import glob
import yaml
import random

def upload_data(config, project):
    image_extension_type = '.jpg'
    label_extension_type = '.txt'

    new_image_path = config['new_image_path']
    new_label_path = config['new_label_path']

    image_glob = glob.glob(new_image_path + '/*' + image_extension_type)
    label_glob = glob.glob(new_label_path+'/*'+ label_extension_type)

    new_data = list(zip(image_glob, label_glob))
    random.shuffle(new_data)
    split_index = int(len(new_data) * 0.8) 

    # 80% train, 20% valid에 업로드
    for i, (image_path, label_path) in enumerate(new_data):
        if i<split_index:
            split = "train"
        else:
            split = "valid"

        project.upload(
            image_path,
            label_path,
            split=split,
        )

