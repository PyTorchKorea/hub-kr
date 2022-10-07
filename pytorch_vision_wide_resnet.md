---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Wide ResNet
summary: Wide Residual Networks
category: researchers
image: wide_resnet.png
author: Sergey Zagoruyko
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
github-id: pytorch/vision
featured_image_1: wide_resnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/Wide_Resnet
---

```python
import torch
# load WRN-50-2:
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
# or WRN-101-2
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 요구하는 미니배치 생성

# GPU 사용이 가능한 경우 속도를 위해 입력과 모델을 GPU로 이동
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# ImageNet 1000개 클래스에 대한 신뢰도 점수를 가진 1000 형태의 Tensor 출력
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

Wide Residual 네트워크는 ResNet에 비해 단순히 채널 수가 증가했습니다. 
그렇지 않으면 아키텍처가 동일합니다. 
병목(bottleneck) 블록이 있는 심층 ImageNet 모델은 내부 3x3 컨볼루션의 채널 수를 증가 시켰습니다.

`wide_resnet50_2` 및 `wide_resnet101_2` 모델은 [Warm Restarts가 있는 SGD(SGDR)](https://arxiv.org/abs/1608.03983)를 사용하여 혼합 정밀도(Mixed Precision) 방식으로 FP16에서 학습되었습니다.
체크 포인트는 크기가 작은 경우 절반 정밀도(batch norm 제외)의 가중치를 가지며 FP32 모델에서도 사용할 수 있습니다.
| Model structure   | Top-1 error | Top-5 error | # parameters |
| ----------------- | :---------: | :---------: | :----------: |
|  wide_resnet50_2  | 21.49       | 5.91        | 68.9M        |
|  wide_resnet101_2 | 21.16       | 5.72        | 126.9M       |

### 참고문헌

 - [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
 - [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
 - [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
