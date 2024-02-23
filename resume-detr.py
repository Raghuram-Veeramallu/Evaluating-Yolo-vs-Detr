from ultralytics import RTDETR
from clearml import Task

task = Task.init(project_name="yolo-detr", task_name="detr-200")

# model = RTDETR("./runs/train4/weights/last.pt")
model = RTDETR("/home/csci8980/veera047/yolo-tests/runs/detect/train4/weights/last.pt")

# Resume training
results = model.train(resume=True)
