# BCP-based Semi-Supervised Medical Image Segmentation

This repository presents my first research project on **semi-supervised learning**, applied to **medical image segmentation**. It includes both a comprehensive **literature survey** and an implementation of a novel approach called **Bidirectional Copy-Paste (BCP)**, built on top of the **Mean Teacher framework**.

> 📄 The full research report can be found in [`FinalReport.pdf`](./FinalReport.pdf)

---

## 🧠 Project Summary

Manual segmentation in medical imaging is expensive and time-consuming, especially in clinical settings. This project aims to:

- Reduce the need for large amounts of labeled data
- Improve segmentation performance using **semi-supervised learning (SSL)**
- Explore and implement the **Mean Teacher + BCP** architecture
- Apply it to the **ACDC** cardiac MRI dataset using U-Net backbone

---

## 🚀 How to Use

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib numpy h5py

### 2. Prepare Dataset 
- Download ACDC Dataset ( link in Link_dataset.txt)
- Update paths inside the notebooks 
