from ultralytics import RTDETR
from PIL import Image

# Load a model
model = RTDETR('/home/csci8980/veera047/yolo-tests/trained_models/detr_14_epochs.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(
    ['/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_006_A_0009.jpg', 
    '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_007_A_0008.jpg',
    '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_2463.jpg',
    '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_007_A_5036.jpg']
)  # return a list of Results objects

# Process results list
for i, r in enumerate(results):
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save(f'./results_{i}.jpg')  # save image
