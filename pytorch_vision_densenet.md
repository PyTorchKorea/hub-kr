---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Densenet
summary: Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion.
category: researchers
image: densenet1.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
github-id: pytorch/vision
featured_image_1: densenet1.png
featured_image_2: densenet2.png
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/Densenet
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=True)
model.eval()
```

사전에 학습된 모든 모델은 동일한 방식으로 정규화된 입력 이미지,
즉, `H` 와 `W` 는 최소 `224` 이상인 `(3 x H x W)` 형태의 3-채널 RGB 이미지의 미니 배치를 요구합니다.
이미지를 `[0, 1]` 범위에서 로드한 다음 `mean = [0.485, 0.456, 0.406]`
과 `std = [0.229, 0.224, 0.225]` 를 통해 정규화합니다.

실행 예시입니다.
```python
# 파이토치 웹사이트에서 예제 이미지 다운로드
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 실행 예시 (torchvision 필요)
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 요구되는 미니배치 생성

# 가능하다면 속도를 위해 입력과 모델을 GPU로 옮깁니다
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# shape이 1000이며 ImageNet의 1000개 클래스에 대한 신뢰도 점수가 있는 텐서
print(output[0])
# 출력에 정규화되지 않은 점수가 있습니다. 확률을 얻으려면 소프트맥스를 실행하세요.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet 레이블 다운로드
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리 읽기
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 이미지 별 Top5 카테고리 조회
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

Dense Convolutional Network (DenseNet)는 순전파(feed-forward) 방식으로 각 레이어를 다른 모든 레이어과 연결합니다. L 계층의 기존 합성곱 신경망이 L개의 연결 - 각 층과 다음 층 사이의 하나 - 인 반면 우리의 신경망은 L(L+1)/2 직접 연결을 가집니다. 각 계층에, 모든 선행 계층의 (feature-map)형상 맵은 입력으로 사용되며, 자체 형상 맵은 모든 후속 계층에 대한 입력으로 사용됩니다. DenseNets는 몇 가지 강력한 장점을 가집니다: 그레디언트가 사라지는 문제를 완화시키고, 특징 전파를 강화하며, 특징 재사용을 권장하며, 매개 변수의 수를 크게 줄입니다.

사전 학습된 모델을 사용한 imagenet 데이터셋의 1-crop 오류율은 다음 표와 같습니다.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  densenet121        | 25.35       | 7.83        |
|  densenet169        | 24.00       | 7.00        |
|  densenet201        | 22.80       | 6.43        |
|  densenet161        | 22.35       | 6.20        |

### 참고 자료

 - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993).
