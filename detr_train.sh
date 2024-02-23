#!/bin/bash -l        
#SBATCH --time=24:00:00
#SBATCH --ntasks=24
#SBATCH --mem=64g
#SBATCH --tmp=64g
#SBATCH --gres=gpu:a100:4
#SBATCH -p a100-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=veera047@umn.edu
module load conda
conda activate yolo
cd yolo-tests/
python detr-run.py
