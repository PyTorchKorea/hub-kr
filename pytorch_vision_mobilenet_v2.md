---
layout: hub_detail
background-class: hub-background
body-class: hub
title: MobileNet v2
summary: Efficient networks optimized for speed and memory, with residual blocks
category: researchers
image: mobilenet_v2_1.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
github-id: pytorch/vision
featured_image_1: mobilenet_v2_1.png
featured_image_2: mobilenet_v2_2.png
accelerator: cuda-optional
order: 10
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model.eval()
```

미리 훈련된(pre-trained) 모든 모델은 입력 이미지를 같은 방식으로 정규화하길 기대합니다.
예를 들어 shape가 `(3 x H x W)`인 3채널 RGB 이미지의 미니배치에서는 `H` 와 `W`가 적어도 `224`이길 기대합니다.
이미지는 `[0, 1]`의 범위에서 읽어들여야 되고, 그 뒤엔 `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`을 이용해서 정규화를 해 줍니다.

다음은 샘플 실행입니다.

```python
# 파이토치 웹사이트에서 예제 이미지를 다운로드 합니다
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 샘플 실행 (torchvision이 필요합니다)
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
input_batch = input_tensor.unsqueeze(0) # 모델의 입력값에 맞춘 미니 배치 생성

# 가능하면 속도를 위해서 입력과 모델을 GPU로 이동 합니다
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Imagenet의 1000개 클래스에 대한 신뢰도 점수가 있는 1000개의 Tensor입니다.
print(output[0])
# 출력에 정규화되지 않은 점수가 있습니다. 확률을 얻으려면 소프트맥스를 사용하면 됩니다.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet 라벨 다운로드
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리 읽기
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 이미지별 최고 확률 카테고리 보여주기
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

MobileNet v2 아키텍처는 역잔차(inverted residual) 구조에 기반을 두고 있습니다. 역잔차 구조는 잔차 블록(residual block)의 입출력이 얇은 병목 계층이 되는 구조입니다. 이러한 계층은 입력에서 확장된 표현형(expanded representation)을 사용하는 전통적인 계층과는 정반대입니다. MobileNet v2는 중간의 확장 계층에서 특징(feature)을 필터링하기 위해 경량화된 깊이 위주의 합성곱(depthwise convolution)을 사용합니다. 추가적으로, 표현 능력(representational power)을 유지하기 위해 좁은 계층 내부의 비선형성은 제거됩니다.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  mobilenet_v2       | 28.12       | 9.71       |


### 참조

 - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
