from ultralytics import RTDETR
from clearml import Task

task = Task.init(project_name="yolo-detr", task_name="detr-new-val")

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = RTDETR('/home/csci8980/veera047/yolo-tests/trained_models/detr_60_epochs.pt')  # load a custom model

# Validate the model
metrics = model.val(
    data="./vdd-yolo.yaml",
)  # no arguments needed, dataset and settings remembered
print(f'map: {metrics.box.map}')    # map50-95
print(f'map50: {metrics.box.map50}')
print(f'map75: {metrics.box.map75}')
print(f'maps: {metrics.box.maps}')
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category
