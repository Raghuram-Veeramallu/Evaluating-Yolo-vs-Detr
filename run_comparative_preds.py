import os
import random
from ultralytics import RTDETR,YOLO
from PIL import Image, ImageDraw
from thop import profile
import torch
from torchvision import transforms

# Function to calculate GFLOPs
def calculate_gflops(model, input_size):
    input = torch.randn(1, *input_size)
    macs, _ = profile(model, inputs=(input, ))
    gflops = macs / 1e9
    return gflops, macs

datafiles = '/home/csci8980/shared/video_diver_dataset/yolo_version/test/test.txt'
datafile_path = '/home/csci8980/shared/video_diver_dataset/yolo_version/test/'

with open(datafiles, 'r') as f:
    img_files = f.readlines()

img_files = list(map(lambda x: x.strip('\n'), img_files))
img_files = list(map(lambda x: x.split('/')[-1], img_files))

# barbados_scuba_006_A_0001
samp = []
img_strt = 10
img_end = 20
for i in range(img_strt, img_end+1):
    samp.append(f'/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_07{i}.jpg')
# samp = random.sample(img_files, 10)
# samp = list(map(lambda x: f'{datafile_path}{x}', samp))

# samp = [
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_007_A_4896.jpg',
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_4284.jpg',
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/pool_flipper_003_A_1057.jpg', 
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/pool_swimmer_004_A_0761.jpg', 
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_007_A_1970.jpg', 
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_2193.jpg', 
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_3442.jpg', 
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_006_A_0718.jpg', 
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_007_A_0108.jpg', 
#   '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_0723.jpg'
# ]

# test_img = Image.open(samp[0])
# convert_tensor = transforms.ToTensor()
# input_size = (1056, 1920)

detr_model_path = '/home/csci8980/veera047/yolo-tests/trained_models/detr_14_best.pt'
yolo_model_path = '/home/csci8980/veera047/yolo-tests/trained_models/best_20_yolo.pt'
img_store_path = '/home/csci8980/veera047/yolo-tests/temporal'

detr_model = RTDETR(detr_model_path)
yolo_model = YOLO(yolo_model_path)

# # Calculate and print GFLOPs
# detr_gflops = calculate_gflops(detr_model, input_size)
# yolo_gflops = calculate_gflops(yolo_model, input_size)
# print(f"DETR Model GFLOPs: {detr_gflops}")
# print(f"YOLO Model GFLOPs: {yolo_gflops}")

detr_results = detr_model(samp)
yolo_results = yolo_model(samp)

samp_names = list(map(lambda x: x.split('/')[-1].split('.')[0], samp))
samp_paths = list(map(lambda x: x.split('.')[0], samp))

# Function to convert YOLO format bounding box to pixel coordinates
def yolo_to_pixels(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return [x1, y1, x2, y2]

for name, r in zip(samp_names, detr_results):
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save(f'{img_store_path}/detr_results_{name}.jpg')  # save image

for name, r in zip(samp_names, yolo_results):
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save(f'{img_store_path}/yolo_results_{name}.jpg')  # save image

for name, p in zip(samp_names, samp):
    img = Image.open(p)
    draw = ImageDraw.Draw(img)

    # Get original annotations
    annotation_path = p.replace('.png', '.txt').replace('.jpg', '.txt')
    with open(annotation_path, 'r') as file:
        annotations = file.readlines()
        
    # Draw original bounding boxes (assuming YOLO format)
    for ann in annotations:
        _, x_center, y_center, width, height = map(float, ann.split())
        bbox = yolo_to_pixels([x_center, y_center, width, height], img.width, img.height)
        draw.rectangle(bbox, outline='red', width=5)  # Original bbox in green
    
    # img_name = os.path.basename(p)
    img.save(f'{img_store_path}/original_{name}.jpg')


# print(samp)
