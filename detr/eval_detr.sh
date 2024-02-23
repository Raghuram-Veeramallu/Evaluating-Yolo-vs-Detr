#!/bin/bash -l        
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32g
#SBATCH --tmp=32g
#SBATCH --gres=gpu:a100:1
#SBATCH -p a100-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=veera047@umn.edu
module load conda
conda activate yolo
cd yolo-tests/detr/
# python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --batch_size 2 --no_aux_loss --eval --resume /home/csci8980/veera047/yolo-tests/detr/checkpoint.pth --coco_path /home/csci8980/shared/video_diver_dataset/yolo_version/
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --batch_size 1 --no_aux_loss --pred --resume /home/csci8980/veera047/yolo-tests/detr/trained_60.pth --coco_path /home/csci8980/shared/video_diver_dataset/yolo_version/ --output_dir /home/csci8980/veera047/yolo-tests/detr/
