---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: HybridNets
summary: HybridNets - 종단간 인식 네트워크
image: hybridnets.jpg
author: Dat Vu Thanh
tags: [vision]
github-link: https://github.com/datvuthanh/HybridNets
github-id: datvuthanh/HybridNets
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
demo-model-link: https://colab.research.google.com/drive/1Uc1ZPoPeh-lAhPQ1CloiVUsOIRAVOGWA
---
## 시작하기 전에

**PyTorch>=1.10**이 설치된 **Python>=3.7** 환경 에서 시작합니다. PyTorch를 설치하려면 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 를 참고하세요. HybridNets 종속 패키지를 설치하려면 아래 명령을 수행해주세요:
```bash
pip install -qr https://raw.githubusercontent.com/datvuthanh/HybridNets/main/requirements.txt  # install dependencies
```

## 모델 설명
 
<img width="100%" src="https://github.com/datvuthanh/HybridNets/raw/main/images/hybridnets.jpg">  

HybridNets는 다중 작업을 위한 종단간 인식 네트워크입니다. 이 다중 네크워크는 교통 물체 감지, 주행 가능 영역 분할 및 차선 감지에 중점을 두었습니다. HybridNets는 임베디드 시스템에서 실시간으로 실행할 수 있으며 BDD100K 데이터셋에서 최신 기술(state-of-the-art)의 수준의 물체 감지, 차선 감지 성능을 보여줍니다.

### 결과

### 교통 물체 감지

|        Model       |  Recall (%)  |   mAP@0.5 (%)   |
|:------------------:|:------------:|:---------------:|
|     `MultiNet`     |     81.3     |       60.2      |
|      `DLT-Net`     |     89.4     |       68.4      |
|   `Faster R-CNN`   |     77.2     |       55.6      |
|      `YOLOv5s`     |     86.8     |       77.2      |
|       `YOLOP`      |     89.2     |       76.5      |
|  **`HybridNets`**  |   **92.8**   |     **77.3**    |

<img src="https://github.com/datvuthanh/HybridNets/raw/main/images/det1.jpg" width="50%" /><img src="https://github.com/datvuthanh/HybridNets/raw/main/images/det2.jpg" width="50%" />
 
### 운전 가능 영역 분할

|       Model      | Drivable mIoU (%) |
|:----------------:|:-----------------:|
|    `MultiNet`    |        71.6       |
|     `DLT-Net`    |        71.3       |
|     `PSPNet`     |        89.6       |
|      `YOLOP`     |        91.5       |
| **`HybridNets`** |      **90.5**     |

<img src="https://github.com/datvuthanh/HybridNets/raw/main/images/road1.jpg" width="50%" /><img src="https://github.com/datvuthanh/HybridNets/raw/main/images/road2.jpg" width="50%" />
 
### 차선 감지

|      Model       | Accuracy (%) | Lane Line IoU (%) |
|:----------------:|:------------:|:-----------------:|
|      `Enet`      |     34.12    |       14.64       |
|      `SCNN`      |     35.79    |       15.84       |
|    `Enet-SAD`    |     36.56    |       16.02       |
|      `YOLOP`     |     70.5     |        26.2       |
| **`HybridNets`** |   **85.4**   |      **31.6**     |

<img src="https://github.com/datvuthanh/HybridNets/raw/main/images/lane1.jpg" width="50%" /><img src="https://github.com/datvuthanh/HybridNets/raw/main/images/lane2.jpg" width="50%" />
  
<img width="100%" src="https://github.com/datvuthanh/HybridNets/raw/main/images/full_video.gif">
 
 
### PyTorch Hub에서 불러오기

이 예제는 사전 훈련된 HybridNets 모델을 불러오고 추론을 위해 이미지를 전달합니다.
```python
import torch

# load model
model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)

#inference
img = torch.randn(1,3,640,384)
features, regression, classification, anchors, segmentation = model(img)
```

### 인용

본 [논문](https://arxiv.org/abs/2203.09035) 과 [코드](https://github.com/datvuthanh/HybridNets) 가 여러분의 연구에 유용하다고 판단되면, GitHub star를 주는 것과 본 논문을 인용하는 것을 고려해 주세요:

```BibTeX
@misc{vu2022hybridnets,
      title={HybridNets: End-to-End Perception Network}, 
      author={Dat Vu and Bao Ngo and Hung Phan},
      year={2022},
      eprint={2203.09035},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
