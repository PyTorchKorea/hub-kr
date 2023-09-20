---
layout: hub_detail
background-class: hub-background
body-class: hub
title: SqueezeNet
summary: 50배 적은 파라미터로 Alexnet 수준의 정확도 제공
category: researchers
image: squeezenet.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py
github-id: pytorch/vision
featured_image_1: squeezenet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/SqueezeNet
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
# 또는
# model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
model.eval()
```

사전에 훈련된 모델은 모두 같은 방식으로 정규화(normalize)한 이미지를 입력으로 받습니다.

예를 들어, `(3 x H x W)` 포맷의 3채널 rgb 이미지들의 미니 배치의 경우 H 와 W 의 크기는 224 이상이어야 합니다.
이 때 모든 픽셀들은 0과 1 사이의 값을 가지도록 변환한 이후 `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` 로 정규화해야 합니다.

실행 예제는 아래와 같습니다.

```python
# pytorch에서 웹사이트에서 예제 이미지 다운로드
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 예제 (토치비전 필요)
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 요구하는 형식인 mini batch 형태로 변환

# 빠르게 실행하기 위해 가능한 경우 model 과 input image 를 gpu 를 사용하도록 설정
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# ImageNet 1000개 범주에 대한 신뢰 점수를 나타내는 텐서 반환
print(output[0])
# 해당 신뢰 점수는 softmax를 취해 확률값으로 변환가능합니다.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet 라벨 다운로드
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 범주 읽기
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 이미지 별로 확률값이 가장 높은 범주 출력
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

`squeezenet1_0` 모델은 [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf) 논문을 구현한 것입니다.

`squeezenet1_1` 모델은 [official squeezenet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1) 에서 왔습니다.
`squeezenet1_0` 수준의 정확도를 유지하며 2.4배 계산이 덜 필요하고, `squeezenet1_0`보다 매개변수의 수가 적습니다.

ImageNet 데이터셋 기준으로 훈련된 모델들의 1-crop 에러율은 아래와 같습니다.

| 모델 | Top-1 에러 | Top-5 에러 |
| --------------- | ----------- | ----------- |
|  squeezenet1_0  | 41.90       | 19.58       |
|  squeezenet1_1  | 41.81       | 19.38       |

### 참조

 - [Squeezenet: Alexnet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf).
