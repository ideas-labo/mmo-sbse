# NAS Dataset Notice
This directory contains the Neural Architecture Search (NAS) experiment code for the MMO-SBSE project.

### Important Note
**We do not provide NAS datasets in this repository**. All required datasets (including CIFAR-10/CIFAR-100, ImageNet, Cityscapes, etc.) are managed by the EvoXBench benchmark suite and must be downloaded separately.

### Dataset Download & Setup
1. Download NAS datasets by following the instructions in the EvoXBench official repository:  
   [https://github.com/EMI-Group/evoxbench](https://github.com/EMI-Group/evoxbench)
2. After downloading, place the datasets (including the `data/` and `database/` subdirectories from EvoXBench) into the `Datasets/` folder under this NAS directory. This ensures compatibility with the MMO-SBSE NAS experiment scripts.