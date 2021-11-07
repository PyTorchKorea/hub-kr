---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ResNext
summary: Next generation ResNets, more efficient and accurate
category: researchers
image: resnext.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
github-id: pytorch/vision
featured_image_1: resnext.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext50_32x4d', pretrained=True)
# or
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnext101_32x8d', pretrained=True)
model.eval()
```

사전 훈련된 모든 모델들은 입력 이미지들이 같은 방식으로 정규화된것으로 생각합니다. 
즉, `(3 x H x W)` 모양의 3채널 RGB 이미지를 말하고 `H` 와 `W`은 각각 최소 `224` 이상을 기대합니다.
이미지는 `[0, 1]` 범위로 로드한 다음 `mean = [0.485, 0.456, 0.406]` 과 `std = [0.229, 0.224, 0.225]`
를 사용하여 정규화를 진행합니다.

Here's a sample execution.
아래는 실행 예시입니다.

```python
# 파이토치 웹사이트에서 예시 이미지를 다운로드합니다.
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 실행 예시 (torchvision이 필요)
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
input_batch = input_tensor.unsqueeze(0) # 모델에 맞는 미니 배치를 만듭니다.

# GPU 사용이 가능하다면 속도를 위해 입력과 모델을 GPU로 이동합니다.
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Imagenet의 1000개 클래스에 대한 confidence scores를 가지는 텐서
print(output[0])
# 출력은 정규화되지 않은 score입니다. 각 클래스에 대한 확률을 얻고 싶다면, softmax를 사용합니다.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet의 label 다운로드
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리 읽기
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# 이미지에 대해 점수가 높은 카테고리들 보여주기
top5_prob, top5_catid = torch.topk(probabilities, 5)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### Model Description

Resnext는 [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)에서 제안된 모델입니다.
2가지 버전의 resnet 모델들이 있는데, 각각 50, 101개의 레이어로 구성되어있습니다.
resnet50과 resnext50의 모델 구조 비교는 Table 1에서 확인할 수 있습니다.
imagenet 데이터셋으로 학습한 사전 훈련 모델들의 Top-1 error 비율은 아래에 적어두었습니다.

|  Model structure  | Top-1 error | Top-5 error |
| ----------------- | ----------- | ----------- |
|  resnext50_32x4d  | 22.38       | 6.30        |
|  resnext101_32x8d | 20.69       | 5.47        |

### References

 - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
