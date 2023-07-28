from roboflow import Roboflow
class roboflow_dataset:
    def __init__(self, version, project):
        self.version = version
        self.project = project
    def load_dataset(self):
        dataset = self.project.version(self.version).download("yolov8")
        return dataset
