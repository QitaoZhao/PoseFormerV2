# PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation

This repo is the official implementation for **PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation**. The paper has been accepted to [CVPR 2023](https://cvpr2023.thecvf.com/).

## Introduction

PoseFormerV2 is built upon [PoseFormer](https://github.com/zczcwh/PoseFormer). It targets improving its efficiency in processing long input sequences and its robustness to noisy 2D joint detection via a frequency-domain joint sequence representation.

**Abstract.** Recently, transformer-based methods have gained significant success in sequential 2D-to-3D lifting human pose estimation. As a pioneering work, PoseFormer captures spatial relations of human joints in each video frame and human dynamics across frames with cascaded transformer layers and has achieved impressive performance. However, in real scenarios, the performance of PoseFormer and its follow-ups is limited by two factors: (a) The length of the input joint sequence; (b) The quality of 2D joint detection. Existing methods typically apply self-attention to all frames of the input sequence, causing a huge computational burden when the frame number is increased to obtain advanced estimation accuracy, and they are not robust to noise natu- rally brought by the limited capability of 2D joint detectors. In this paper, we propose PoseFormerV2, which exploits a compact representation of lengthy skeleton sequences in the frequency domain to efficiently scale up the receptive field and boost robustness to noisy 2D joint detection. With minimum modifications to PoseFormer, the proposed method effectively fuses features both in the time domain and frequency domain, enjoying a better speed-accuracy trade-off than its precursor. Extensive experiments on two benchmark datasets (i.e., Human3.6M and MPI-INF-3DHP) demonstrate that the proposed approach significantly outperforms the original PoseFormer and other transformer-based variants.

![PoseFormerV2](./images/framework.jpg)

<img src="./images/improvements.jpg" alt="PoseFormerV2" style="zoom:7%;" />

## Visualizations

![PoseFormerV2](./images/visualization.jpg)

![PoseFormerV2](./images/noise_comparison.jpg)

## Cite PoseFormerV2

If you find PoseFormerV2 useful in your research, please consider citing:

```bibtex
@inproceedings{
	title={PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation},
	author={Zhao, Qitao and Zheng, Ce and Liu, Mengyuan and Wang, Pichao, and Chen, Chen},
	booktitle={Conference on Computer Vision and Pattern Recognition 2023},
	year={2023},
}
```

## Environment

The code is developed and tested under the following environment

- Python 3.8
- PyTorch 1.11.0
- CUDA 11.3

## Usage

### Dataset preparation

Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset as follows:

```
code_root/
└── data/
	├── data_2d_h36m_gt.npz
	├── data_2d_h36m_cpn_ft_h36m_dbb.npz
	└── data_3d_h36m.npz
```

### Training

You can train PoseFormerV2 on a single GPU with the following command:

```bash
python run_poseformer.py -g 0 -k cpn_ft_h36m_dbb -frame 27 -frame-kept 3 -coeff-kept 3 -c checkpoint/NAMED_PATH
```

This example shows how to train PoseFormerV2 with 3 central frames and 3 DCT coefficients from a 27-frame sequence.

### Evaluation

You can evaluate PoseFormerV2 with prepared checkpoints as:

```bash
python run_poseformer.py -g 0 -k cpn_ft_h36m_dbb -frame 27 -frame-kept 3 -coeff-kept 3 -c checkpoint/NAMED_PATH --evaluate best_epoch.bin
```

## Acknowledgment

Our codes are mainly based on [PoseFormer](https://github.com/zczcwh/PoseFormer). Many thanks to the authors!

