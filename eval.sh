#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                             # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1                                  # Ask for 1 GPU
#SBATCH --mem=64G           
#SBATCH --time=24:00:00                                    
#SBATCH --output=./slurm_logs/eval/output-%j.txt
#SBATCH --error=./slurm_logs/eval/error-%j.txt 



module load miniconda/3
conda init
conda activate aro

# Evaluation on Classification tasks
# python -m clip_benchmark.cli eval \
#     --batch_size 64 \
#     --model_type sail \
#     --model dinov2b_nv2 \
#     --pretrained $PRETRAINED \
#     --task "zeroshot_classification"  \
#     --dataset wds/food101 cifar10 cifar100 sun397 cars fgvc_aircraft dtd pets caltech101 flowers  \
#     --dataset_root "clip_benchmark_datasets/{dataset}" \
#     --output "{dataset}_sharelock_12m_{model}_{language}_{task}.json"


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
    --pretrained laion400m_e32 \
    --dataset mscoco_captions flickr30k \
    --dataset_root "clip_benchmark_datasets/{dataset}" \
    --output "results/{task}/{dataset}/vitb32_dl3m_{model}.json"
