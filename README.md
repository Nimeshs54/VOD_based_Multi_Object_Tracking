# vod_based_tracking_nimesh

## Introduction

Multi-Object Tracking (MOT) is a fundamental computer vision task that encompasses the detection and tracking of objects in video sequences. Tracktor, an advanced MOT algorithm, employs a s Faster R-CNN object detector. However, the inherent complexities of tracking multiple objects in video sequences present significant challenges, notably the degradation of object appearances due to factors such as rapid motion, camera defocus, and pose variations.

In this repository, we propose a novel approach to enhance multi-object tracking performance by integrating Sequence Level Semantics Aggregation (SELSA), a Video Object Detection (VID) technique, into the Tracktor framework. SELSA is employed as a substitute for Faster R-CNN in our project to address the aforementioned challenges, thereby improving tracking accuracy and robustness. We test the proposed method on the MOT challenge dataset and achieve a notable 3.6% enhancement in Multiple Object Tracking Accuracy (MOTA) and a 1.2% increase in Higher Order Tracking Accuracy (HOTA) over the conventional Tracktor with Faster R-CNN.

## Installation

Please refer to [Installation.md](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/install.md) for installation.

## Data Preparation

Please download the MOTChallenge dataset from [here](https://motchallenge.net/) and for setup the data please refer to [this](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/dataset.md).

## Result

Result of Tracktor with FRCNN and SELSA on a validation set of MOT15 dataset.

|                      | Precision | Recall | IDF1 | MOTA | HOTA |
| -------------------- | --------- | ------ | ---- | ---- | ---- |
| Tracking with FRCNN  | 85.2      | 81.6   | 68.3 | 66.6 | 52.9 |
| Tracking with SELSA  | 86.7      | 85.0   | 64.9 | 70.5 | 53.2 |
