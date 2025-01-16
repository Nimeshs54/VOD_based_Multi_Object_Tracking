# Video Object Detection-Based Tracking

## Introduction

Multi-Object Tracking (MOT) is a fundamental computer vision task involving the detection and tracking of objects in video sequences. **Tracktor**, an advanced MOT algorithm, leverages a Faster R-CNN object detector. However, tracking multiple objects in video sequences poses challenges, including:

- Degradation of object appearances due to rapid motion.
- Camera defocus.
- Pose variations.

In this repository, we propose a novel approach to enhance multi-object tracking performance by integrating **Sequence Level Semantics Aggregation (SELSA)**—a Video Object Detection (VID) technique—into the Tracktor framework. SELSA replaces Faster R-CNN in our pipeline to address these challenges, improving tracking accuracy and robustness. Our method, tested on the **MOTChallenge dataset**, achieves:

- A **3.6% enhancement** in Multiple Object Tracking Accuracy (MOTA).
- A **1.2% increase** in Higher Order Tracking Accuracy (HOTA) compared to the conventional Tracktor with Faster R-CNN.

---

## Installation

Follow the installation guide provided by [MMTracking Installation](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/install.md).

---

## Data Preparation

1. Download the **MOTChallenge dataset** from [MOTChallenge.net](https://motchallenge.net/).
2. Refer to the [Data Setup Documentation](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/dataset.md) for proper dataset preparation.

---

## Results

### Performance Comparison on the MOT15 Validation Dataset

| Method               | Precision | Recall | IDF1 | MOTA | HOTA |
| -------------------- | --------- | ------ | ---- | ---- | ---- |
| **Tracking with FRCNN** | 85.2      | 81.6   | 68.3 | 66.6 | 52.9 |
| **Tracking with SELSA** | 86.7      | 85.0   | 64.9 | 70.5 | 53.2 |

---

This repository showcases the potential of combining SELSA with the Tracktor framework to push the boundaries of multi-object tracking performance.
