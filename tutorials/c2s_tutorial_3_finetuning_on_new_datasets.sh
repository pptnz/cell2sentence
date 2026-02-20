#!/bin/bash
#SBATCH --job-name=c2s_tutorial_3_finetuning_on_new_datasets
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=50G

source /oak/stanford/groups/amnewman/mkang9/util/miniconda3/etc/profile.d/conda.sh
conda activate cell2sentence

date
python c2s_tutorial_3_finetuning_on_new_datasets.py
date

jupyter nbconvert --to script c2s_tutorial_4_cell_type_prediction.ipynb
python c2s_tutorial_4_cell_type_prediction.py

date
python ../evaluate_pred.py
python ../benchmark.py
date