# Collective Migration-Inspired Large-Deformation Compensation for Nonrigid Image Registration

This repository contains the official implementation of our IJCV paper "Collective Migration-Inspired Large-Deformation Compensation for Nonrigid Image Registration".

**Please give a star and cite if you find this repo useful.**

## Introduction

Inspired by the collective nature of animal migration, we introduce a novel perspective on image registration by framing large-deformation compensation as collective manifold. In this perspective, we regard salient points in images as leaders, similar to leader animals in a migratory flock, playing a pivotal role in guiding the overall registration process. In contrast, non-salient points are considered as flocks, akin to the animals in the flock following the leaders. To estimate a collective manifold, we divide large-deformation registration into three parts following the migration pattern of animals: routes decision based on collectiveness quantification, routes execution with collectiveness maintenance, and cascaded migration with collectiveness inheritance. These three parts are integrated in the collective cascaded migration (CCM) framework, effectively compensating for image large-scale deformation.

![](.\Figures\flowchart.jpg)

## Implementation Notes

- Hybrid Python/MATLAB implementation (requires `matlab_engine`)

## Datasets

We assess the effectiveness of our method using six diverse datasets. The initial pair comprises 2D semi-synthetic datasets, wherein synthetic deformations are introduced to abdominal slice images. The subsequent four datasets are sourced from challenging open datasets, encompassing three 2D datasets (Fundus Image Registration dataset- [FIRE](https://projects.ics.forth.gr/cvrl/fire/), Object-Kinect- [OK](https://www.verlab.dcc.ufmg.br/descriptors/), Object-Simulation- [OS](https://www.verlab.dcc.ufmg.br/descriptors/)) and one 3D dataset (Lung CT- [LUNG](https://learn2reg.grand-challenge.org/Datasets/)). All datasets exhibit substantial large-scale deformations, visually depicted in the figure below.

![](.\Figures\dataset.jpg)

The 2D semi-synthetic datasets, NS and NRS, can be downloaded at this [link](https://mega.nz/file/cylUHL4B#UyVdn7g8T2qhVPoseWcBuYKb4o25F_kLPnXZOgXWArY).

Please download the full datasets from their official sources:
- [FIRE Dataset](https://projects.ics.forth.gr/cvrl/fire/)
- [OK Dataset](https://www.verlab.dcc.ufmg.br/descriptors/) 
- [OS Dataset](https://www.verlab.dcc.ufmg.br/descriptors/)
- [LUNG Dataset](https://learn2reg.grand-challenge.org/Datasets/)

Due to GitHub storage limitations, the `Data/` folder only contains:
- Subsampled test datasets

## Quick Start

### 1. Set up environment

```cmd
conda env create -f ccm.yaml
conda activate ccm
```

### 2. Run demo scripts (partial test data)

Run the Jupyter notebook files in `Scripts/` to test all modules with sample datasets.

## Results

### Qualitative results

![](.\Figures\object results OS.jpg)

![](.\Figures\lung.jpg)

### Performance boosting

![](.\Figures\fire table.png)

![](.\Figures\lung table.png)

## Contact

If you have any problem, please contact us via [migyangz@gmail.com](mailto:migyangz@gmail.com) or [xiaowuga@gmail.com](mailto:xiaowuga@gmail.com). We greatly appreciate everyone's feedback and insights. Please do not hesitate to get in touch!

### Todo

- Pure Python + GPU accelerated code version is coming soon!

## Citation

Please consider citing our work if you find it useful:

```
TO BE DONE
```