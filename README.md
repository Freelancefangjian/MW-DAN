# MW-DAN
MW-DAN: Multilevel Wavelet-Deep Aggregation Network for Hyperspectral Image Super-Resolution

MW-DAN (Multilevel Wavelet-Deep Aggregation Network) is the deep learning model implemented in this repository for the Hyperspectral Image (HSI) and Multispectral Image (MSI) fusion super-resolution task. This work has been published in Acta Electronica Sinica, Issue 1, 2024.

📖 Abstract

Utilizing the fusion of low spatial resolution hyperspectral images (LR-HSI) and high spatial resolution multispectral images (HR-MSI) to achieve hyperspectral spatial resolution enhancement is a significant research area. Current deep learning methods face challenges in effectively mining complementary information, injecting spatial details, and maintaining spectral fidelity.

To address this, we propose a Multilevel Wavelet-Deep Aggregation Network (MW-DAN). This network innovatively combines the Undecimated Wavelet Transform (UDWT) with a deep residual network, constructing a dual-branch complementary information fusion framework:

1.  Spatial Detail Extraction Module
2.  Deep Residual Aggregation Module

The network is trained end-to-end with LR-HSI, HR-MSI, and HR-HSI to learn an excellent spatial-spectral fusion nonlinear mapping. Extensive experiments on the CAVE and Harvard simulation datasets, as well as the real Hyperion-Sentinel-2 dataset, demonstrate that MW-DAN outperforms mainstream methods in multiple objective evaluation metrics, spectral preservation capability, and visual quality.

✨ Key Features & Contributions

•   Novel Network Architecture: Proposes MW-DAN, which employs a dual-branch end-to-end learning framework. It injects spatial high-frequency information through wavelet directional subbands layer by layer, effectively improving fusion performance while reducing the network's parameter scale.

•   Efficient Aggregation Module: Designs a Deep Residual Aggregation (DRA) module incorporating skip-layer aggregation connections. This shortens the backpropagation path, mitigates gradient vanishing, enhances feature reuse and information aggregation capabilities, and achieves efficient complementary fusion of spatial and spectral information.

•   Comprehensively Validated Performance: Conducts extensive simulation and real-data experiments on multiple public datasets. Ablation studies verify the effectiveness of each component (e.g., number of DRA modules, loss function, wavelet type, skip connections). MW-DAN achieves leading performance on key metrics such as MPSNR, SAM, and ERGAS.

🎯 Method Overview

The overall architecture of MW-DAN is illustrated in the figure below (refer to Figure 1 in the paper):
1.  Input: HR-MSI and upsampled LR-HSI.
2.  Spatial Detail Extraction.
3.  Feature Fusion and Reconstruction.
4.  Deep Residual Aggregation Module.

📊 Datasets & Performance

We validated the model's performance on the following datasets:

Dataset Description Usage

CAVE 32 indoor scenes, 512×512×31 Train/Test

Harvard 50 indoor/outdoor scenes, ~1000×1000×31 Train/Test

Hyperion-Sentinel-2 Real satellite data for cross-sensor fusion Real-data Validation
Partial Quantitative Results on the CAVE Dataset (Average):

---

| Method | MPSNR | SAM | ERGAS | MSSIM |
|--------|-------|-----|-------|-------|
| HySure | 34.78 | 11.55 | 2.42 | 0.9107 |
| NSSR | 40.32 | 5.04 | 1.26 | 0.9809 |
| DHSIS | 46.30 | 3.85 | 0.66 | 0.9905 |
| DBIN | 45.78 | 3.60 | 0.68 | 0.9926 |
| MoG-DCN | 46.32 | 3.60 | 0.65 | 0.9924 |
| MW-DAN (Ours) | 46.86 | 3.39 | 0.62 | 0.9931 |

---

Note: The table above shows partial results reported in the paper. For complete experimental results (including the Harvard dataset, real data, and ablation studies), please refer to the original paper.

🚀 Quick Start

1. Environment Dependencies

This project is implemented based on TensorFlow 2. The experimental environment in the paper is described below. You can also use a similar configuration.
•   Paper's Experimental Environment: Windows 10, Intel(R) Core(TM) i7-9700 CPU, NVIDIA 2080Ti GPU, TensorFlow 2.x.

•   Recommended Setup:
    Python >= 3.7
    TensorFlow >= 2.4.0 (Recommended to match your CUDA/cuDNN version)
    numpy
    scipy
    matplotlib
    tqdm
    
    (Note: A requirements.txt file will be provided in the code repository.)

2. Installation & Data Preparation

1.  Clone the Repository
    git clone https://github.com/your_username/MW-DAN.git
    cd MW-DAN
    
2.  Install Dependencies
    pip install -r requirements.txt
    
3.  Prepare Data
    ◦   Download the http://www1.cs.columbia.edu/CAVE/databases/multispectral/ and https://vision.seas.harvard.edu/hyperspec/ datasets.

    ◦   Preprocess the data according to the description in Section 3.2 of the paper (spatial downsampling, spectral response simulation, patching, etc.). We provide a preprocessing script prepare_data.py.

3. Train the Model

Train the MW-DAN model on the CAVE dataset (default: 2 DRA modules, átrous wavelet, L1 loss).
python train.py --dataset CAVE --data_path ./data/CAVE --save_path ./checkpoints --epochs 1000 --lr 0.0001

Parameter Notes: Training uses the Adam optimizer (β1=0.9, β2=0.999), batch size 32, as detailed in Section 3.1 of the paper.

4. Test & Evaluation

Evaluate the trained model on the test set and compute metrics like MPSNR, SAM, ERGAS, MSSIM.
python test.py --dataset CAVE --data_path ./data/CAVE/test --model_path ./checkpoints/best_model.h5


5. Predict on Your Own Hyperspectral Data

We provide a simple prediction script to fuse your own LR-HSI and HR-MSI data.
python predict.py --lrhsi_path ./your_data/lr.hdr --hrmsi_path ./your_data/msi.tif --model_path ./checkpoints/best_model.h5 --output_path ./result.hdr


📁 Project Structure


MW-DAN/
├── data/               # Data directory (needs to be downloaded and preprocessed)
├── src/
│   ├── models/
│   │   └── mw_dan.py  # MW-DAN model definition (TensorFlow/Keras)
│   ├── utils/
│   │   ├── dataset.py # Data loading and preprocessing
│   │   ├── wavelet.py # Undecimated Wavelet Transform implementation
│   │   └── metrics.py # Evaluation metrics calculation
│   ├── train.py       # Model training script
│   ├── test.py        # Model testing and evaluation script
│   └── predict.py     # Single/Batch data prediction script
├── configs/           # Configuration files
├── checkpoints/       # Model checkpoint directory
├── results/           # Experimental results and image output
├── requirements.txt   # Project dependencies
└── README.md          # This file


📜 Citation

If you find our work helpful for your research, please cite our paper (Chinese reference format as requested):

```bibtex
@article{fang2024mwdan,
  title={MW-DAN: Multilevel Wavelet-Deep Aggregation Network for Hyperspectral Image Super-Resolution},
  author={FANG Jian, YANG Jing-xiang, XIAO Liang},
  journal={Acta Electronica Sinica},
  volume={52},
  number={1},
  pages={201--216},
  year={2024},
  doi={10.12263/DZXB.20220800}
}
```

📄 License

This project is licensed under the LICENSE.

⁉️ Contact

For any questions or suggestions, please feel free to contact:
•   Author: Jian Fang

•   Email: [Your email, e.g., fangjian@njust.edu.cn]

Acknowledgments: This work was supported by the National Natural Science Foundation of China (No.61871226, No.61571230, No.62001226), the Key R&D Plan of Jiangsu Province (No.BE2018727), and other funding projects.
