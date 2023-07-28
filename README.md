# Smarthome with Hand Gesture

> 네이버 부스트캠프 AI tech 5기 CV lv3 5조 Final프로젝트 공간입니다.

<img src="https://i.ibb.co/tD3GMWq/2.png">

## Contributors

|강희성 |                                                  김영한|김정현 |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [<img src="https://avatars.githubusercontent.com/u/90888774?v=4" alt="" style="width:100px;100px;">](https://github.com/atom1905h) <br/> | [<img src="https://avatars.githubusercontent.com/u/50921080?v=4" alt="" style="width:100px;100px;">](https://github.com/dkdlel6887) <br/> | [<img src="https://avatars.githubusercontent.com/u/114405449?v=4" alt="" style="width:100px;100px;">](https://github.com/Jhyuny) <br/>

<br></br>

# 프로젝트 소개

```sh
 제스처 인식을 통해 카메라 내의 기기를 조작합니다.
```
![시연영상](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-05/assets/90888774/a94d6e43-5736-4e04-a97b-f484e50fdf8b)

## 사용 방법

```sh
크롬 웹캠 보안 허용하기

1. chrome://flags/#unsafely-treat-insecure-origin-as-secure 접속
2. Insecure origins treated 항목에 (http://101.101.209.25:30003/) 기입  >>> uri 알맞게 변경
3. disable -> enabled 로 변경
```
<img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-05/assets/50921080/0fd95122-51a8-428c-803f-71c72cf506a1">


## 개발 환경 설정
```sh
python 3.8.5
GPU NVIDIA Tesla V100 
```

## 실행 방법
```sh
pip install -r requirements.txt

# 학습 시 roboflow api key를 secret.yaml 파일에 기입
python train.py  # roboflow 내 최신 버전의 데이터셋 다운로드 후 학습하여 best.pt 파일 생성

# pretrained best.pt 파일 이용 시
python main.py  # 개인 로컬에서 웹캠을 통해 실행
python service.py  # ip 및 port 변경하여 실행하면 로컬에서 웹페이지를 통해 실행 가능
```

## 기능 설명
```sh
입력 받은 영상 내 물체 인식 후 → 손동작으로 on/off 및 볼륨, 온도 수치 조절하기  
추가 하고 싶은 기기 등록 → 새로운 데이터셋 구축 및 모델 자동 학습 후 물체 인식하여 조작
```

## 프로세스
<img src=https://github.com/boostcampaitech5/level3_cv_finalproject-cv-05/assets/90888774/07e83583-3eab-4c3e-88d0-4831ff9b00b0 width="700" height="300"/>

## 사용 기술
<img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-05/assets/50921080/a92371d6-8ffd-48e8-885a-959a8ad268a5" width="700" height="300"/>

## 사용 데이터셋
<img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-05/assets/50921080/d74cedf1-af1f-4191-84fd-28b7edd7e3bc" width="700" height="300"/>

```sh
- 모델 학습에 사용한 기본 데이터셋입니다. 가정에서 흔히 볼 수 있는 물체로 선택
- 클래스 균형을 위해 각각 300장씩 모아 roboflow를 통해 annotation을 진행
```
## 손동작 종류
<img src="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-05/assets/50921080/4459410a-d3f1-4906-bf5a-a89afbe73099" width="700" height="200"/>

## 참고 링크
<a href="https://youtu.be/YDPDhL6tOs0"><img src="https://img.shields.io/badge/Presentation(Video)-000000?style=flat-square&logo=youtube&logoColor=fc2403"/></a>  
<a href="https://github.com/boostcampaitech5/level3_cv_finalproject-cv-05/files/12190578/CV_5._.pdf"><img src="https://img.shields.io/badge/Presentation(Pdf)-000000?style=flat-square&logo=googledrive&logoColor=03fc07"/></a>  
<a href="https://www.notion.so/boostcampait/CV-05-Smarthome-with-Hand-Gesture-70e4f7a5335847fcb380c66611f5e74d?pvs=4"><img src="https://img.shields.io/badge/Notion-000000?style=flat-squrare&logo=Notion"/></a>  

## Commit Type

- feat : 새로운 기능 추가, 기존의 기능을 요구 사항에 맞추어 수정
- fix : 기능에 대한 버그 수정
- build : 빌드 관련 수정
- chore : 패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore
- ci : CI 관련 설정 수정
- docs : 문서(주석) 수정
- style : 코드 스타일, 포맷팅에 대한 수정
- refactor : 기능의 변화가 아닌 코드 리팩터링 ex) 변수 이름 변경
- test : 테스트 코드 추가/수정
- release : 버전 릴리즈
--------------
