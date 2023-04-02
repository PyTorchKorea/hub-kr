---
layout: hub_detail
background-class: hub-background
body-class: hub
title: IBN-Net
summary: Networks with domain/appearance invariance
category: researchers
image: ibnnet.png
author: Xingang Pan
tags: [vision]
github-link: https://github.com/XingangPan/IBN-Net
github-id: XingangPan/IBN-Net
featured_image_1: ibnnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/IBN-Net
---

```python
import torch
model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
model.eval()
```

모든 사전 훈련된 모델은 동일한 방식으로 정규화된 입력 이미지를 요구합니다.
즉, `H`와 `W`가 최소 `224`의 크기를 가지는 `(3 x H x W)`형태의 3채널 RGB 이미지의 미니배치가 필요합니다. 
이미지를 [0, 1] 범위로 불러온 다음 `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`를 이용하여 정규화해야 합니다.

다음은 실행예시입니다.

```python
# 파이토치 웹 사이트에서 예제 이미지 다운로드
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 실행예시 (torchvision이 요구됩니다.)
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
input_batch = input_tensor.unsqueeze(0) #  모델에서 요구하는 미니배치 생성

# GPU 사용이 가능한 경우 속도를 위해 입력과 모델을 GPU로 이동
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# ImageNet의 1000개 클래스에 대한 신뢰도 점수를 가진 1000 형태의 Tensor 출력
print(output[0])
# 출력은 정규화되어있지 않습니다. 소프트맥스를 실행하여 확률을 얻을 수 있습니다.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet 레이블 다운로드
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리 읽어오기
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 이미지마다 상위 카테고리 5개 보여주기
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

IBN-Net은 도메인/외관 불변성을 갖는 CNN 모델입니다.
Style transfer에 영감을 얻어 IBN-Net은 단일 심층 네트워크에서 인스턴스 정규화와 일괄 정규화를 신중하게 통합합니다.
모델 복잡성을 추가하지 않고 모델링 및 범용성을 모두 증가시키는 간단한 방법을 제공합니다. 
IBN-Net은 특히 교차 도메인 또는 사람/차량 재식별 작업에 적합합니다.

ImageNet 데이터셋을 사용했을 때 사전 훈련된 모델들의 정확도는 다음과 같습니다.

| Model name | Top-1 acc   | Top-5 acc   |
| --------------- | ----------- | ----------- |
| resnet50_ibn_a  | 77.46       | 93.68       |
| resnet101_ibn_a | 78.61       | 94.41       |
| resnext101_ibn_a | 79.12      | 94.58       |
| se_resnet101_ibn_a | 78.75    | 94.49       |

두 가지 Re-ID 벤치마크 Market1501 및 DukeMTMC-reID에 대한 rank1/mAP는 아래에 나열되어 있습니다.([michuanhaohao/reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)에서 가져왔습니다.)

| Backbone | Market1501 | DukeMTMC-reID |
| --- | -- | -- |
| ResNet50 | 94.5 (85.9) | 86.4 (76.4) |
| ResNet101 | 94.5 (87.1) |  87.6 (77.6) |
| SeResNet50 | 94.4 (86.3) | 86.4 (76.5) |
| SeResNet101 | 94.6 (87.3) | 87.5 (78.0) |
| SeResNeXt50 | 94.9 (87.6) | 88.0 (78.3) |
| SeResNeXt101 | 95.0 (88.0) | 88.4 (79.0) |
| ResNet50-IBN-a | 95.0 (88.2) | 90.1 (79.1) |

### 참고문헌

 - [Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net](https://arxiv.org/abs/1807.09441)
