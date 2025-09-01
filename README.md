# Prompt Transformer for Generalized Category Discovery

<p align="center">
    <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Less_Attention_is_More_Prompt_Transformer_for_Generalized_Category_Discovery_CVPR_2025_paper.html"><img src="https://img.shields.io/badge/-CVPR%202025-68488b"></a>
  <a href="https://github.com/CVMI-Lab/SlotCon/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>
<p align="center">
	Less Attention is more: Prompt Transformer for Generalized Category Discovery (CVPR 2025)<br>
  By
  Wei Zhang, 
  <a href="https://faculty.bjtu.edu.cn/8221/">Baopeng Zhang</a>*, 
  <a href="https://faculty.bjtu.edu.cn/8902/">Zhu Teng</a>,
  Wenxin Luo, 
  Junnan Zou,
  Jianping Fan
</p>


This repository is the official implementation of AptGCD: [Less Attention is more: Prompt Transformer for Generalized Category Discovery](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Less_Attention_is_More_Prompt_Transformer_for_Generalized_Category_Discovery_CVPR_2025_paper.pdf)
![overview](assets/overview.png)
## Running
### Dependencies
The code was trained on python3.10 pytorch2.0.0 and CUDA11.7.
You can install requirements by pip ```install -r requirements.txt```.

### Config
Set paths to datasets and desired log directories in ```config.py```.

### Datasets
* We trained on three generic datasets: [CIFAR-10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html), [CIFAR-100](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html), [ImageNet-100](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet).
* We also user four fine-grained benchmarks: [StanfordCars](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb), [CUB-200](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb), [FGVC-Aircraft](https://github.com/sgvaze/osr_closed_set_all_you_need#ssb) and [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6).

### Scripts
Train the model 

## Acknowledge
The code is built on this repo: https://github.com/Cliffia123/LegoGCD. Many thanks to their briliant works.