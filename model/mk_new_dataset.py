import os
import copy
from random import *
from PIL import Image
from rembg import remove


def remove_bg():
    uimg_path = "사용자로 부터 받은 사진 디렉터리 경로"
    rembg_path = "./rembg_images/"
    if not os.path.exists(rembg_path):
        os.mkdir(rembg_path)

    l = len(os.listdir(uimg_path))
    for i in range(l):
        img = Image.open(f"{uimg_path}/{str(i)}.jpg")  # 기존 사진 불러오기
        out = remove(img)  # 배경 제거하기

        # 변경된 이미지 저장하기
        out.save(rembg_path + str(i) + ".png")


# remove_bg

k = 0  # 0: 배경 o  1: 배경 x

# 사용자로부터 받은 이미지 4장
if k == 0:
    user_images = os.listdir("물체 사진 디렉터리 경로")
elif k == 1:
    user_images = os.listdir("배경이 제거된 물체 사진 디렉터리 경로")

# 배경으로 사용할 이미지 (약 30장)
bg_images = os.listdir("배경사진이 있는 디렉터리 경로")


path = "./new_dataset"  # 현재 경로에 new_dataset 디렉터리 추가
if not os.path.exists(path):
    os.mkdir(path)

img_path = f"{path}/bg_user_images"  # 새로운 이미지 저장할 디렉토리
label_path = f"{path}/bg_user_labels"  # 새로운 레이블 저장할 디렉토리
cnt = 0  # 저장할 이미지 이름
class_name = "mouse"  # 사용자가 추가한 클래스 이름

# resize를 진행할때 특정 ratio(r)를 곱해서 변형
for bg in bg_images:
    bg_img = Image.open(path + "/bg_images/" + bg)  # image open
    bg_img = bg_img.resize((640, 640))
    bw, bh = bg.size
    for user in user_images:
        if k == 0:
            uimg = Image.open(path + "/user_images/" + user)
        elif k == 1:
            uimg = Image.open(path + "/rembg_images/" + user)  # 배경제거 이미지 사용 시
        w, h = uimg.size  # user image size 전체 동일할 경우 고정
        e = 1e-2
        m = min(bw / w, bh / h)
        r = uniform(e, m / 2)
        rw, rh = int(r * w), int(r * h)
        uimg = uimg.resize((rw, rh))  # resize 적용

        # bg에 붙이기
        xmin = randint(0, bw - rw)
        ymin = randint(0, bh - rh)
        xcen = xmin + rw / 2
        ycen = ymin + rh / 2
        bg_user_img = copy.deepcopy(bg_img)
        bg_user_img.paste(uimg, (xmin, ymin), True)  # xmin, ymin 좌표에 resize된 uimg 붙임

        # label.txt 저장
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        label = f"{class_name} {xcen/bw} {ycen/bh} {rw/bw} {rh/bh}"
        file = open(f"{label_path}/{str(cnt)}.txt", "w")
        file.write(label)
        file.close()

        # image.jpg 저장
        if not os.path.exists(img_path):
            os.mkdir(f"{img_path}")

        if k == 0:
            bg_user_img.save(img_path + "/" + str(cnt) + ".jpg")
            print(f"{cnt}.jpg, {cnt}.txt 저장 완료")
        elif k == 1:
            bg_user_img.save(img_path + "/" + str(cnt) + ".png")  # 배경제거 이미지 사용 시
            print(f"{cnt}.png, {cnt}.txt 저장 완료")

        cnt += 1
