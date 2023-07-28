from roboflow import Roboflow
import glob
import yaml
import random

def upload_data(config, project):
    image_extension_type = ['.jpg','.png']
    label_extension_type = '.txt'

    new_image_path = config['new_image_path']
    new_label_path = config['new_label_path']
    image_glob=[]
    for extension_type in image_extension_type:
        image_glob.extend(glob.glob(new_image_path + '/*' + extension_type))
    label_glob = glob.glob(new_label_path+'/*'+ label_extension_type)
    image_glob.sort()
    label_glob.sort()
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

