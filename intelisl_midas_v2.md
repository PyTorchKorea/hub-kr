---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: MiDaS
summary: MiDaS models for computing relative depth from a single image.
image: intel-logo.png
author: Intel ISL
tags: [vision]
github-link: https://github.com/intel-isl/MiDaS
github-id: intel-isl/MiDaS
featured_image_1: midas_samples.png
featured_image_2: no-image
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/MiDaS
---

### 모델 설명

[MiDaS](https://arxiv.org/abs/1907.01341)는 단일 이미지로부터 상대적 역 깊이(relative inverse depth)를 계산합니다. 본 저장소는 작지만 고속의 모델부터 가장 높은 정확도를 제공하는 매우 큰 모델까지 다양한 사례를 다루는 여러 모델을 제공합니다. 또한 모델은 광범위한 입력에서 높은 품질을 보장하기 위해 다목적(multi-objective) 최적화를 사용해 10개의 개별 데이터 셋에 대해 훈련되었습니다. 

### 종속 패키지 설치

MiDas 모델은 [timm](https://github.com/rwightman/pytorch-image-models)을 사용합니다 아래 명령어를 통해 설치해 주세요.
```shell
pip install timm
```

### 사용 예시

파이토치 홈페이지로부터 이미지를 다운로드합니다.
```python
import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
```

모델을 로드합니다. (개요는 [https://github.com/intel-isl/MiDaS/#Accuracy](https://github.com/intel-isl/MiDaS/#Accuracy) 를 참조하세요.)
```python
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
```
GPU 사용이 가능한 환경이라면, 모델에 GPU를 사용합니다.
```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
```


여러가지 모델에 입력할 이미지를 크기 변경(resize)이나 정규화(normalize)하기 위한 변환(transform)을 불러옵니다.
```python
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
```

이미지를 로드하고 변환(transforms)을 적용합니다.
```python
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)
```


기존 해상도로 예측 및 크기 변경합니다.
```python
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
```

결과 출력
```python
plt.imshow(output)
# plt.show()
```

### 참고문헌
[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341)

[Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)

만약 MiDaS 모델을 사용한다면 본 논문을 인용해 주세요:
```bibtex
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```
```bibtex
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```
