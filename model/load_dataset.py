from roboflow import Roboflow
class roboflow_dataset:
    def __init__(self, version):
        self.version = version
    def load_dataset(self):
        rf = Roboflow(api_key="I56H1gcAvLeqblSIcWF9")
        project = rf.workspace("final-projectboostcamp-ai-tech").project("final-project-dataset-r1wxc")
        dataset = project.version(self.version).download("yolov8")
        return dataset
