from ultralytics import YOLO
from clearml import Task

task = Task.init(project_name="yolo-detr", task_name="yolo-200-val")

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/home/csci8980/veera047/yolo-tests/trained_models/best_20_yolo.pt')  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(f'map: {metrics.box.map}')    # map50-95
print(f'map50: {metrics.box.map50}')
print(f'map75: {metrics.box.map75}')
print(f'maps: {metrics.box.maps}')
