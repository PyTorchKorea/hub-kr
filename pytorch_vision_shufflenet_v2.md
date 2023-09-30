---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ShuffleNet v2
summary: ImageNet에서 사전 훈련된 속도와 메모리에 최적화된 효율적인 ConvNet
category: researchers
image: shufflenet_v2_1.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
github-id: pytorch/vision
featured_image_1: shufflenet_v2_1.png
featured_image_2: shufflenet_v2_2.png
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/ShuffleNet_v2
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
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
# Imagnet의 1000개 클래스에 대한 신뢰도 점수를 가진 1000 형태의 텐서 출력
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

이전에는 신경망 아키텍처 설계는 주로 FLOP와 같은 계산 복잡성의 간접 측정 기준에 따라 진행되었습니다. 그러나 속도와 같은 직접적인 측정은 메모리 액세스 비용 및 플랫폼 특성과 같은 다른 요소에도 의존합니다. 일련의 통제된 실험을 기반으로, 이 작업은 효율적인 네트워크 설계를 위한 몇 가지 실용적인 지침을 도출합니다. 따라서 ShuffleNet V2라는 새로운 아키텍처가 제시됩니다. 조건 변화에 따른 모델 평가를 통해 속도와 정확도 트레이드오프 측면에서 최고 수준임을 확인했습니다.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  shufflenet_v2       | 30.64       | 11.68       |


### 참고문헌

 - [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
