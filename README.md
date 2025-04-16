
# [CVPR 2025] SAIL: Swift Alignment of Image and Language Evaluation Code

[![Paper](https://img.shields.io/badge/paper-arxiv.2412.03561-B31B1B.svg)](https://arxiv.org/abs/2412.04616)
[![SAIL](https://img.shields.io/badge/Project-Page-FFD700?style=for-the-badge?logo=flag)](https://lezhang7.github.io/sail.github.io/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-SAIL-FFD700?logo=huggingface&logoColor=yellow)](https://huggingface.co/le723z/sail/tree/main)

## Evaluation

This repository contains the evaluation code for the SAIL project, built upon [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark).

### How to Run Evaluation

To evaluate SAIL models, simply run:

```bash
python -m clip_benchmark.cli eval \
    --batch_size 64 \
    --model_type sail \
    --model dinov2b_nv2 \
    --pretrained $PRETRAINED \
    --task "zeroshot_classification"  \
    --dataset wds/food101 cifar10 cifar100 sun397 cars fgvc_aircraft dtd pets caltech101 flowers  \
    --dataset_root "clip_benchmark_datasets/{dataset}" \
    --output "{dataset}_sharelock_12m_{model}_{language}_{task}.json"
```

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{zhang2024assessing,
  title={Assessing and Learning Alignment of Unimodal Vision and Language Models},
  author={Zhang, Le and Yang, Qian and Agrawal, Aishwarya},
  journal={arXiv preprint arXiv:2412.04616},
  year={2024}
}
```

## Acknowledgements

This evaluation framework is based on [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark). We thank the authors for their excellent work.
