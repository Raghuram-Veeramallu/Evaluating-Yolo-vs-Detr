from thop import profile
import torch
from ultralytics import RTDETR

model = RTDETR("/home/csci8980/veera047/yolo-tests/trained_models/detr_16_epochs.pt")

# input = 

macs, params = profile(models, inputs=(input, ))
gflops = macs / 1e9

print(f'GFLOPS: {gflops}')
