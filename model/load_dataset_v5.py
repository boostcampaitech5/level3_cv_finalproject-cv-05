from roboflow import Roboflow

# api key, project name, version 사용자에 맞게 변경
rf = Roboflow(api_key="I56H1gcAvLeqblSIcWF9")
project = rf.workspace("final-projectboostcamp-ai-tech").project("final-project-dataset-keqzz")
dataset = project.version(6).download("yolov5")
