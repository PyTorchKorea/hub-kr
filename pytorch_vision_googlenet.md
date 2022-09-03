---
layout: hub_detail
background-class: hub-background
body-class: hub
title: GoogLeNet
summary: GoogLeNet was based on a deep convolutional neural network architecture codenamed "Inception" which won ImageNet 2014.
category: researchers
image: googlenet1.png
author: Pytorch Team
tags: [vision]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
github-id: pytorch/vision
featured_image_1: googlenet1.png
featured_image_2: googlenet2.png
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/GoogleNet
order: 10
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()
```

모든 사전 훈련된 모델들은 입력 이미지가 동일한 방식으로 정규화되었다고 상정합니다.
즉, 미니 배치(mini-batch)의 3-채널 RGB 이미지들은 `(3 x H x W)`의 형태를 가지며, 해당 `H`와 `W`는 최소 `224` 이상이어야 합니다.
각 이미지는 `[0, 1]`의 범위 내에서 로드되어야 하며, `mean = [0.485, 0.456, 0.406]` 과 `std = [0.229, 0.224, 0.225]`을 이용해 정규화되어야 합니다.
다음은 실행 예제 입니다.

```python
# 파이토치 웹 사이트에서 다운로드한 이미지 입니다.
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 예시 코드 (torchvision 필요)
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 가정하는대로 미니배치 생성

# gpu를 사용할 수 있다면, 속도를 위해 입력과 모델을 gpu로 옮김
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# output은 shape가 [1000]인 Tensor 자료형이며, 이는 Imagenet 데이터셋의 1000개의 각 클래스에 대한 모델의 확신도(confidence)를 나타냅니다.
print(output[0])
# output은 정규화되지 않았으므로, 확률화하기 위해 softmax 함수를 처리합니다.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet 데이터셋 레이블 다운로드
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리(클래스) 읽기
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 각 이미지에 대한 top 5 카테고리 출력
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```


### 모델 설명

GoogLeNet은 코드네임 "Inception"으로 불리는 신경망 아키텍처에 기반한 깊은 합성곱 신경망입니다. 이 모델은 ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014) 에서 새로운 SOTA(state of the art)를 달성했습니다. 사전훈련된 모델로 ImageNet 데이터셋에서의 단일-크롭 방식으로 오류 비율을 측정한 결과는 아래와 같습니다.

| 모델 구조 | Top-1 오류 | Top-5 오류 |
| --------------- | ----------- | ----------- |
|  googlenet       | 30.22       | 10.47       |



### 참고문헌

 - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)