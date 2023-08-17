---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: YOLOP
summary: YOLOP pretrained on the BDD100K dataset
image: yolop.png
author: Hust Visual Learning Team
tags: [vision]
github-link: https://github.com/hustvl/YOLOP
github-id: hustvl/YOLOP
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/YOLOP
---

## 시작하기 전 참고 사항
YOLOP 종속 패키지를 설치하려면 아래 명령을 수행해주세요:
```bash
pip install -qr https://github.com/hustvl/YOLOP/blob/main/requirements.txt  # install dependencies
```


## YOLOP: You Only Look Once for Panoptic driving Perception

### 모델 설명

<img width="800" alt="YOLOP Model" src="https://github.com/hustvl/YOLOP/raw/main/pictures/yolop.png">
&nbsp;

- YOLOP는 자율 주행에서 중요한, 다음의 세 가지 작업을 공동으로 처리할 수 있는 효율적인 다중 작업 네트워크 입니다. 이 다중 네트워크는 물체 감지(object detection), 주행 영역 분할(drivable area segmentation), 차선 인식(lane detection)을 수행합니다. 또한 YOLOP는  **BDD100K** 데이터셋에서 최신 기술(state-of-the-art)의 수준을 유지하면서 임베디드 기기에서 실시간성에 도달한 최초의 모델입니다.


### 결과

#### 차량 객체(Traffic Object) 인식 결과

| Model          | Recall(%) | mAP50(%) | Speed(fps) |
| -------------- | --------- | -------- | ---------- |
| `Multinet`     | 81.3      | 60.2     | 8.6        |
| `DLT-Net`      | 89.4      | 68.4     | 9.3        |
| `Faster R-CNN` | 77.2      | 55.6     | 5.3        |
| `YOLOv5s`      | 86.8      | 77.2     | 82         |
| `YOLOP(ours)`  | 89.2      | 76.5     | 41         |

#### 주행 가능 영역 인식 결과

| Model         | mIOU(%) | Speed(fps) |
| ------------- | ------- | ---------- |
| `Multinet`    | 71.6    | 8.6        |
| `DLT-Net`     | 71.3    | 9.3        |
| `PSPNet`      | 89.6    | 11.1       |
| `YOLOP(ours)` | 91.5    | 41         |

#### 차선 인식 결과

| Model         | mIOU(%) | IOU(%) |
| ------------- | ------- | ------ |
| `ENet`        | 34.12   | 14.64  |
| `SCNN`        | 35.79   | 15.84  |
| `ENet-SAD`    | 36.56   | 16.02  |
| `YOLOP(ours)` | 70.50   | 26.20  |

#### 조건 변화에 따른 모델 평가 1 (Ablation Studies 1): End-to-end v.s. Step-by-step

| Training_method | Recall(%) | AP(%) | mIoU(%) | Accuracy(%) | IoU(%) |
| --------------- | --------- | ----- | ------- | ----------- | ------ |
| `ES-W`          | 87.0      | 75.3  | 90.4    | 66.8        | 26.2   |
| `ED-W`          | 87.3      | 76.0  | 91.6    | 71.2        | 26.1   |
| `ES-D-W`        | 87.0      | 75.1  | 91.7    | 68.6        | 27.0   |
| `ED-S-W`        | 87.5      | 76.1  | 91.6    | 68.0        | 26.8   |
| `End-to-end`    | 89.2      | 76.5  | 91.5    | 70.5        | 26.2   |

#### 조건 변화에 따른 모델 평가 2 (Ablation Studies 2): Multi-task v.s. Single task

| Training_method | Recall(%) | AP(%) | mIoU(%) | Accuracy(%) | IoU(%) | Speed(ms/frame) |
| --------------- | --------- | ----- | ------- | ----------- | ------ | --------------- |
| `Det(only)`     | 88.2      | 76.9  | -       | -           | -      | 15.7            |
| `Da-Seg(only)`  | -         | -     | 92.0    | -           | -      | 14.8            |
| `Ll-Seg(only)`  | -         | -     | -       | 79.6        | 27.9   | 14.8            |
| `Multitask`     | 89.2      | 76.5  | 91.5    | 70.5        | 26.2   | 24.4            |

**안내**:

- 표 4에서 E, D, S, W는 인코더(Encoder), 검출 헤드(Detect head), 2개의 세그먼트 헤드(Segment heads) 와 전체 네트워크를 의미합니다. 그래서 이 알고리즘(이 알고리즘은 첫째, 인코더 및 검출 헤드만 학습합니다. 그 후, 인코더 및 검출 헤드를 고정하고 두 개의 분할(segmentation) 헤드를 학습합니다. 마지막으로, 전체 네트워크는 세 가지 작업 모두에 대해 함께 학습됩니다.)은 ED-S-W로 표기되며, 다른 알고리즘도 마찬가지입니다.

### 시각화

#### 차량 객체(Traffic Object) 인식 결과

<img width="800" alt="Traffic Object Detection Result" src="https://github.com/hustvl/YOLOP/raw/main/pictures/detect.png">
&nbsp;

#### 주행 가능 영역 인식 결과

<img width="800" alt="Drivable Area Segmentation Result" src="https://github.com/hustvl/YOLOP/raw/main/pictures/da.png">
&nbsp;

#### 차선 인식 결과

<img width="800" alt="Lane Detection Result" src="https://github.com/hustvl/YOLOP/raw/main/pictures/ll.png">
&nbsp;

**안내**:

- 차선 인식의 시각화 결과는 이차함수 형태로 근사하는 과정(quadratic fitting)을 통해 후처리(post processed) 되었습니다.

### 배포

YOLOP 모델은 이미지 캡쳐를 **Zed Camera**가 장착된 **Jetson Tx2**에서 실시간으로 추론할 수 있습니다. 속도 향상을 위해 **TensorRT**를 사용합니다. 모델의 배포와 추론을 위해 [github code](https://github.com/hustvl/YOLOP/tree/main/toolkits/deploy) 에서 코드를 제공합니다.


### 파이토치 허브로부터 모델 불러오기
이 예제는 사전에 학습된 **YOLOP** 모델을 불러오고 추론을 위한 이미지를 모델에 전달합니다.
```python
import torch

model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

img = torch.randn(1,3,640,640)
det_out, da_seg_out,ll_seg_out = model(img)
```

### 인용(Citation)

See for more detail in [github code](https://github.com/hustvl/YOLOP) and [arxiv paper](https://arxiv.org/abs/2108.11250).

본 [논문](https://arxiv.org/abs/2108.11250) 과 [코드](https://github.com/hustvl/YOLOP) 가 여러분의 연구에 유용하다고 판단되면, GitHub star를 주는 것과 본 논문을 인용하는 것을 고려해 주세요:

```BibTeX
@article{wu2022yolop,
  title={Yolop: You only look once for panoptic driving perception},
  author={Wu, Dong and Liao, Man-Wen and Zhang, Wei-Tian and Wang, Xing-Gang and Bai, Xiang and Cheng, Wen-Qing and Liu, Wen-Yu},
  journal={Machine Intelligence Research},
  pages={1--13},
  year={2022},
  publisher={Springer}
}
```