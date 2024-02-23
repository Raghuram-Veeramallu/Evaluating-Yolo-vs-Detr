#!/bin/bash -l        
#SBATCH --time=24:00:00
#SBATCH --ntasks=24
#SBATCH --mem=32g
#SBATCH --tmp=32g
#SBATCH --gres=gpu:a100:4
#SBATCH -p a100-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=veera047@umn.edu
module load conda
conda activate yolo
cd yolo-tests/detr/
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --epochs 100 --coco_path /home/csci8980/shared/video_diver_dataset/yolo_version/ --resume /home/csci8980/veera047/yolo-tests/detr/trained_60.pth --output_dir /home/csci8980/veera047/yolo-tests/detr/ --start_epoch 60
