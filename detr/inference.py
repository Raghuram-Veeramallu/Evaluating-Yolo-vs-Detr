import argparse
from PIL import Image
import torch
from models.detr import DETR
import matplotlib.pyplot as plt
from models.backbone import build_backbone
from models.transformer import build_transformer

# Create the parser
parser = argparse.ArgumentParser(description='Parse model configuration')

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to the pretrained model. If set, only the mask head will be trained")
# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')

# * Segmentation
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

# dataset parameters
parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', type=str)
parser.add_argument('--coco_panoptic_path', type=str)
parser.add_argument('--remove_difficult', action='store_true')

parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

state_dict = torch.load('/home/csci8980/veera047/yolo-tests/detr/trained_60.pth', map_location='cuda:0')

# backbone = build_backbone(**{'backbone': 'resnet50', 'dilation': True, 'position_embedding': 'sine', 'lr_backbone': 1e-5})
# transformer = build_transformer(**{'enc_layers': 6, 'dec_layers': 6, 'dim_feedforward': 2048, 'hidden_dim': 256,
#                                 'dropout': 0.1, 'nheads': 8, 'num_queries': 100, 'pre_norm': True})

# args = parser.parse_args()

# backbone = build_backbone(args)
# transformer = build_transformer(args)

model_state_dict = state_dict['model']

args = state_dict['args']

backbone = build_backbone(args)
transformer = build_transformer(args)

num_classes = 20 if args.dataset_file != 'coco' else 91
if args.dataset_file == "coco_panoptic":
    # for panoptic, we just add a num_classes that is large enough to hold
    # max_obj_id + 1, but the exact value doesn't really matter
    num_classes = 250
device = torch.device(args.device)

detr = DETR(backbone, transformer, num_classes=num_classes, num_queries=args.num_queries, aux_loss=args.aux_loss)
detr.load_state_dict(model_state_dict)

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# import datasets.transforms as T
import torchvision.transforms as T
import random
import torchvision.transforms.functional as F

def resize(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    return rescaled_image

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = random.choice(self.sizes)
        return resize(img, size, self.max_size)

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Resize((512, 512)),
                T.Compose([
                    RandomResize([400, 500, 600]),
                    # T.RandomSizeCrop(384, 600),
                    RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            RandomResize([800], max_size=1333),
            # T.Resize([800]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

import numpy as np
import util.misc as utils

pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).to('cuda:0').view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).to('cuda:0').view(3, 1, 1)
normalizer = lambda x: (x - pixel_mean) / pixel_std

def detect(im, model, transform):

    img = transform(im)

    img = [normalizer(img.to('cuda:0'))]

    img = utils.NestedTensor(img, None)

    model.to('cuda:0')
    # img = img.to('cuda:0')

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

    return probas[keep], bboxes_scaled

# image = Image.open('/content/img2/training/image_2/000979.png')
# image = np.asarray(image)
# image = image.astype('float32')
# x = torch.from_numpy(image)
# x = x.to(device)
# output = model_without_ddp(x)

def plot_results(pil_img, prob, boxes, name):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax) in prob, boxes.tolist():
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='r', linewidth=5))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='red'))
    plt.axis('off')
    plt.savefig(f'/home/csci8980/veera047/yolo-tests/inference/detr_orig_{name}.jpg')

samps = [
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_007_A_4896.jpg',
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_4284.jpg',
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/pool_flipper_003_A_1057.jpg', 
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/pool_swimmer_004_A_0761.jpg', 
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_007_A_1970.jpg', 
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_2193.jpg', 
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_3442.jpg', 
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_006_A_0718.jpg', 
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_007_A_0108.jpg', 
  '/home/csci8980/shared/video_diver_dataset/yolo_version/test/barbados_scuba_011_D_0723.jpg'
]

detr.eval()
transforms = make_coco_transforms('val')
for each in samps:
    im = Image.open(each)
    scores, boxes = detect(im, detr, transforms)
    plot_results(im, scores, boxes)
