---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Semi-supervised and semi-weakly supervised ImageNet Models
summary: Billion scale semi-supervised learning for image classification 논문에서 제안된 ResNet, ResNext 모델
category: researchers
image: ssl-image.png
author: Facebook AI
tags: [vision]
github-link: https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/blob/master/hubconf.py
github-id: facebookresearch/semi-supervised-ImageNet1K-models
featured_image_1: ssl-image.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/semi-supervised-ImageNet1K-models
---

```python
import torch

# === 해시태그된 9억 4천만개의 이미지를 활용한 Semi-weakly supervised 사전 학습 모델 ===
model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
# ================= YFCC100M 데이터를 활용한 Semi-supervised 사전 학습 모델 ==================
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_ssl')
# model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_ssl')
model.eval()
```

사전에 학습된 모든 모델은 동일한 방식으로 정규화된 입력 이미지, 즉, `H` 와 `W` 는 최소 `224` 이상인 `(3 x H x W)` 형태의 3-채널 RGB 이미지의 미니 배치를 요구합니다. 이미지를 `[0, 1]` 범위에서 불러온 다음 `mean = [0.485, 0.456, 0.406]` 과 `std = [0.229, 0.224, 0.225]`를 통해 정규화합니다.

실행 예시입니다.

```python
# 파이토치 웹사이트에서 예제 이미지를 다운로드합니다.
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 실행 예시입니다. (torchvision 필요)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # 모델에서 요구하는 미니배치를 생성합니다.

# 가능하다면 속도를 위해 입력과 모델을 GPU로 옮깁니다.
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# 1000개의 ImageNet 클래스에 대한 신뢰도 점수(confidence score)를 가진 1000 크기의 Tensor
print(output[0])
# output엔 정규화되지 않은 신뢰도 점수가 있습니다. 확률값을 얻으려면 softmax를 실행하세요.
print(torch.nn.functional.softmax(output[0], dim=0))

```

### 모델 설명
본 문서에선 [Billion-scale Semi-Supervised Learning for Image Classification](https://arxiv.org/abs/1905.00546)에서 제안된 Semi-supervised, Semi-weakly supervised 방식의 ImageNet 분류 모델을 다룹니다.

"Semi-supervised" 방식에서 대용량(hight-capacity)의 teacher 모델은 ImageNet1K 훈련 데이터로 학습됩니다. student 모델은 레이블이 없는 YFCC100M의 일부 이미지를 활용해 사전 학습하며, 이후 ImageNet1K의 훈련 데이터를 통해서 파인 튜닝합니다. 자세한 사항은 앞서 언급한 논문에서 확인할 수 있습니다.

"Semi-weakly supervised" 방식에서 teacher 모델은 해시태그가 포함된 9억 4천만장의 이미지 일부를 활용해 사전 학습되며, 이후 ImageNet1K 훈련 데이터로 파인 튜닝됩니다. 활용된 해시태그는 1500개 정도이며 ImageNet1K 레이블의 동의어 집합(synsets)들을 모은 것입니다. 해시태그는 teacher 모델 사전 학습 과정에서만 레이블로 활용됩니다. student 모델은 teacher 모델이 사용한 이미지와 ImageNet1k 레이블로 사전 학습하며, 이후 ImageNet1K의 훈련 데이터를 통해서 파인 튜닝합니다.

[Xie *et al*.](https://arxiv.org/pdf/1611.05431.pdf), [Mixup](https://arxiv.org/pdf/1710.09412.pdf), [LabelRefinery](https://arxiv.org/pdf/1805.02641.pdf), [Autoaugment](https://arxiv.org/pdf/1805.09501.pdf), [Weakly supervised](https://arxiv.org/pdf/1805.00932.pdf) 기법을 활용했을 때와 비교했을 때, Semi-supervised 및 Semi-weakly-supervised 방식은 ResNet, ResNext 모델의 ImageNet Top-1 검증 정확도를 크게 개선했습니다. 예시, **ResNet-50 구조로 ImageNet 검증 정확도를 81.2% 기록했습니다.**.


| Architecture       |   Supervision   | #Parameters | FLOPS | Top-1 Acc. | Top-5 Acc. |
| ------------------ | :--------------:|:----------: | :---: | :--------: | :--------: |
| ResNet-18          | semi-supervised        |14M     | 2B   |     72.8      | 91.5    |
| ResNet-50          | semi-supervised        |25M     | 4B   |     79.3      | 94.9    |
| ResNeXt-50 32x4d   | semi-supervised        |25M     | 4B   |     80.3      | 95.4    |
| ResNeXt-101 32x4d  | semi-supervised        |42M     | 8B   |     81.0      | 95.7    |
| ResNeXt-101 32x8d  | semi-supervised        |88M     | 16B   |     81.7    |  96.1   |
| ResNeXt-101 32x16d | semi-supervised        |193M    | 36B   |     81.9   | 96.2     |
| ResNet-18          | semi-weakly supervised |14M     | 2B   |    **73.4**    |  91.9      |
| ResNet-50          | semi-weakly supervised |25M     | 4B   |    **81.2**    |  96.0      |
| ResNeXt-50 32x4d   | semi-weakly supervised |25M     | 4B   |    **82.2**    |  96.3      |
| ResNeXt-101 32x4d  | semi-weakly supervised |42M     | 8B   |    **83.4**    |  96.8      |
| ResNeXt-101 32x8d  | semi-weakly supervised |88M     | 16B   |  **84.3**    |  97.2    |
| ResNeXt-101 32x16d | semi-weakly supervised |193M    | 36B   |  **84.8**    |  97.4    |


## 인용

저장소에 공개된 모델을 사용할 땐, 다음 논문을 인용해주세요. ([Billion-scale Semi-Supervised Learning for Image Classification](https://arxiv.org/abs/1905.00546))
```
@misc{yalniz2019billionscale,
    title={Billion-scale semi-supervised learning for image classification},
    author={I. Zeki Yalniz and Hervé Jégou and Kan Chen and Manohar Paluri and Dhruv Mahajan},
    year={2019},
    eprint={1905.00546},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
