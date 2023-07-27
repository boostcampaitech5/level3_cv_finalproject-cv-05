import os
import copy
import yaml
from random import *
from PIL import Image
from rembg import remove


def remove_bg(uimg_path, rembg_path):  # 사용자가 추가한 물체 이미지 디렉터리 경로 입력받음
    if not os.path.exists(rembg_path):
        os.mkdir(rembg_path)

    l = len(os.listdir(uimg_path))
    for i in range(l):
        img = Image.open(f"{uimg_path}/{str(i)}.jpg")  # 기존 사진 불러오기
        out = remove(img)  # 배경 제거하기

        # 변경된 이미지 저장하기
        out.save(rembg_path + "/" + str(i) + ".png")


def paste_image(class_name, bg_path, bg_images, uimg_path, user_images, image_path, label_path, removed):
    cnt = 0  # 저장할 이미지 이름

    # resize를 진행할때 특정 ratio(r)를 곱해서 변형
    for bg in bg_images:
        bg_img = Image.open(bg_path + "/" + bg)  # image open
        bg_img = bg_img.resize((640, 640))
        bw, bh = bg_img.size

        for user in user_images:
            uimg = Image.open(uimg_path + "/" + user)

            w, h = uimg.size
            e = 1e-2
            m = min(bw / w, bh / h)
            r = uniform(e, m / 2)
            rw, rh = int(r * w), int(r * h)
            uimg = uimg.resize((rw, rh))  # resize 적용

            # bg에 붙이기
            xmin = randint(0, bw - rw)
            ymin = randint(0, bh - rh)
            xmax = xmin + rw
            ymax = ymin + rh
            bg_user_img = copy.deepcopy(bg_img)
            if not removed:
                bg_user_img.paste(uimg, (xmin, ymin))  # xmin, ymin 좌표에 resize된 uimg 붙임
            if removed:
                bg_user_img.paste(uimg, (xmin, ymin), uimg)

            # label.txt 저장
            if not os.path.exists(label_path):
                os.mkdir(label_path)
            label = f"{class_name} {xmin/bw} {ymin/bh} {xmax/bw} {ymax/bh}"
            file = open(f"{label_path}/{str(cnt)}.txt", "w")
            file.write(label)
            file.close()

            # image 저장
            if not os.path.exists(image_path):
                os.mkdir(f"{image_path}")
            if not removed:
                bg_user_img.save(image_path + "/" + str(cnt) + ".jpg")
                print(f"{cnt}.jpg, {cnt}.txt 저장 완료")
            elif removed:
                bg_user_img.save(image_path + "/" + str(cnt) + ".png")  # 배경제거 이미지 사용 시
                print(f"{cnt}.png, {cnt}.txt 저장 완료")

            cnt += 1


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, yaml_path):
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def make_dir(path):
     if not os.path.exists(path):
        os.mkdir(path)

def main(removed):
    # path 정리
    paths = "../model/new_dataset"  # 현재 경로에 new_dataset 디렉터리 추가
    make_dir(paths)

    uimg_path = paths + "/user_images"  # 사용자 물체 이미지 저장 경로
    rembg_path = paths + "/rembg_images"  # uimg에서 배경 제거한 이미지 저장 경로
    bg_path = paths + "/bg_images"  # 배경이미지 저장 경로
    make_dir(uimg_path)
    make_dir(rembg_path)
    make_dir(bg_path)
    image_path = paths + "/bg_user_images"  # 새로운 이미지 저장할 디렉토리
    label_path = paths + "/bg_user_labels"  # 새로운 레이블 저장할 디렉토리
    rem_image_path = paths + "/rembg_user_images"  # 새로운 이미지 저장할 디렉토리
    rem_label_path = paths + "/rembg_user_labels"  # 새로운 레이블 저장할 디렉토리
    make_dir(image_path)
    make_dir(label_path)
    make_dir(rem_image_path)
    make_dir(rem_label_path)
    yaml_path = "../model/config.yaml"  # config 수정
    config = load_config(yaml_path)
    if removed:
        config["new_image_path"] = rem_image_path
        config["new_label_path"] = rem_label_path
    if not removed:
        config["new_image_path"] = image_path
        config["new_label_path"] = label_path
    save_config(config, yaml_path)

    # 배경으로 사용할 이미지 (약 30장)
    bg_images = os.listdir(bg_path)  # 배경사진이 있는 디렉터리 경로로 변경

    # 사용자로부터 받은 이미지 4장
    if not removed:
        user_images = os.listdir(uimg_path)
        paste_image(config["class_name"], bg_path, bg_images, uimg_path, user_images, image_path, label_path, removed)
    elif removed:
        remove_bg(uimg_path, rembg_path)
        user_images = os.listdir(rembg_path)
        paste_image(
            config["class_name"], bg_path, bg_images, rembg_path, user_images, rem_image_path, rem_label_path, removed
        )


if __name__ == "__main__":
    removed = True  # True # False: 물체 배경유지  True: 물체 배경제거
    main(removed)
