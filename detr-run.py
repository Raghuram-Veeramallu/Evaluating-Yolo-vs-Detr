from ultralytics import RTDETR
from clearml import Task

task = Task.init(project_name="yolo-detr", task_name="detr-200")

# model = RTDETR("rtdetr-l.pt")
# model = RTDETR("/home/csci8980/veera047/yolo-tests/runs/detect/train4/weights/last.pt")
model = RTDETR("/home/csci8980/veera047/yolo-tests/trained_models/detr_16_epochs.pt")

def freeze_layer(trainer):
    model = trainer.model
    freeze = [f'model.{x}.' for x in range(300)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
            print(f"{num_freeze} layers are freezed.")

model.add_callback("on_train_start", freeze_layer)
model.train(
    data="./vdd-yolo.yaml",
    epochs=30,
    batch=64,
    save=True,
    imgsz=640,
    device=[0, 1, 2, 3],
    resume=True,
)
metrics = model.val()
path = model.export(format="onnx")
