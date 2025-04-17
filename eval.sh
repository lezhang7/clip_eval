#!/bin/bash
#SBATCH --job-name=clip_eval
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=128G           
#SBATCH --time=1:00:00                                    
#SBATCH --output=./slurm_logs/colxlip/job_output-%j.txt
#SBATCH --error=./slurm_logs/colxlip/job_error-%j.txt 




source ~/.bashrc
source ~/vlm/bin/activate


# python -m clip_benchmark.cli eval \
#     --batch_size 64 \
#     --model_type colxlip \
#     --model ViT-B-16-colxlip \
#     --task=zeroshot_retrieval \
#     --pretrained /home/mila/l/le.zhang/scratch/colxlip/src/logs/2025_04_08-01_14_39-model_ViT-B-16-colxlip-lr_5e-06-b_196-j_8-p_amp/checkpoints/epoch_1.pt \
#     --dataset mscoco_captions flickr30k \
#     --dataset_root "clip_benchmark_datasets/{dataset}" \
#     --output "{dataset}_dreamclip_vitb_12m_{model}_{language}_{task}.json"

## baseline
python -m clip_benchmark.cli eval \
    --batch_size 64 \
    --model ViT-B-32 \
    --task=zeroshot_retrieval \
    --pretrained /home/z/zhangle7/links/scratch/colxlip/src/logs/vitb32-ft-baseline-bs256/checkpoints/epoch_5.pt \
    --dataset flickr30k mscoco_captions  \
    --dataset_root "/home/z/zhangle7/links/scratch/clip_eval/clip_benchmark_datasets/{dataset}" \
    --output "results/{task}/{dataset}/vitb32_dl3m_{model}.json"
