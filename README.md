# DLKUNet: Lightweight and Efficient Network with Depthwise Large Kernel for Multi-Organ Segmentation

[![Paper](https://img.shields.io/badge/Paper-Wiley-blue)](https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.70035)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Accurate multi-organ segmentation is crucial in computer-aided diagnosis, surgical navigation, and radiotherapy. This repository provides the official implementation of **DLKUNet**, a lightweight and efficient deep learning framework for medical image segmentation.

---

## 🔥 Highlights

* **Lightweight & Efficient**: Uses depthwise large kernel convolutions for effective multi-scale feature extraction.
* **Flexible Models**: Provides **DLKUNet-S**, **DLKUNet-M**, and **DLKUNet-L** for different trade-offs between accuracy and speed.
* **Novel Training Strategy**: Designed to work seamlessly with the proposed architecture to boost performance.
* **Extensive Validation**: Evaluated on **Synapse multi-organ segmentation** and **ACDC cardiac segmentation** datasets.

---

## 📊 Main Results

* On **Synapse dataset**:

  * **DLKUNet-L** achieves **13.89 mm HD95**, with only **65% parameters** of Swin-Unet.

* On **ACDC dataset**:

  * **DLKUNet-S** achieves **91.71% Dice** using only **4.5% parameters** of Swin-Unet.
  * **DLKUNet-M** achieves **91.74% Dice** using only **16.52% parameters** of Swin-Unet.

These results demonstrate DLKUNet’s superior balance of **accuracy, efficiency, and practicality**.

---

## 📂 Repository Structure

```
DLKUNet/
├── models/           # Network architectures (DLKUNet-S, M, L)
├── datasets/         # Data loading scripts (Synapse, ACDC)
├── train.py          # Training script
├── test.py           # Testing & evaluation script
├── utils/            # Helper functions
└── README.md         # Project description
```

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/University-Pro/DLKUNet.git
cd DLKUNet
```

### 2. Create Environment

```bash
conda create -n dlkunet python=3.8 -y
conda activate dlkunet
pip install -r requirements.txt
```

### 3. Dataset Preparation

* **Synapse**: Download from [official Synapse website](https://www.synapse.org/).
* **ACDC**: Download from [ACDC challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/).

Update the dataset paths in `config/` before training.

### 4. Train

```bash
python train.py --dataset Synapse --model DLKUNet-L --output ./results/synapse
```

### 5. Test

```bash
python test.py --dataset ACDC --model DLKUNet-S --checkpoint ./checkpoints/dlkunet_s.pth
```

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2025dlkunet,
  title={DLKUNet: Lightweight and Efficient Network with Depthwise Large Kernel for Multi-Organ Segmentation},
  author={Your Name and Others},
  journal={IMA Journal of Applied Mathematics},
  year={2025},
  publisher={Wiley},
  doi={10.1002/ima.70035}
}
```

---

## 📌 Links

* 📄 [Paper (Wiley)](https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.70035)
* 💻 [GitHub Repository](https://github.com/University-Pro/DLKUNet)

---

## 📜 License

This project is released under the [MIT License](LICENSE).

要不要我帮你把 README 再加上 **模型结构图/结果可视化图** 的示例？这样会更吸引 GitHub 用户。
