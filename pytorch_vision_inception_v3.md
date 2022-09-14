---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Inception_v3
summary: Also called GoogleNetv3, a famous ConvNet trained on Imagenet from 2015
category: researchers
image: inception_v3.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py
github-id: pytorch/vision
featured_image_1: inception_v3.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/Inception_v3
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()
```

사전 훈련된 모델들을 사용할 때는 동일한 방식으로 정규화된 이미지를 입력으로 넣어야 합니다.
즉, 미니 배치(mini-batch)의 3-채널 RGB 이미지들은 `(3 x H x W)`의 형태를 가지며, 해당 `H`와 `W`는 최소 `224` 이상이어야 합니다.
각 이미지는 `[0, 1]`의 범위 내에서 불러와야 하며, `mean = [0.485, 0.456, 0.406]` 과 `std = [0.229, 0.224, 0.225]`을 이용해 정규화되어야 합니다.
다음은 실행 예제 입니다.

```python
# 파이토치 웹 사이트에서 이미지 다운로드
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
    transforms.Resize(299),
    transforms.CenterCrop(299),
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
# output은 shape가 [1000]인 Tensor 자료형이며, 이는 Imagenet 데이터셋의 1000개의 각 클래스에 대한 모델의 확신도(confidence)를 나타냄
print(output[0])
# output은 정규화되지 않았으므로, 확률화하기 위해 softmax 함수를 처리
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

Inception v3는 합성곱 연산을 적절히 분해하고 적극적인 정규화를 통해 추가된 계산을 가능한 한 효율적으로 활용하는 것을 목표로 네트워크를 확장하는 방법에 대한 탐색을 기반으로 합니다. ILSVRC 2012 (ImageNet) 분류 문제에서 본 논문은 당시 기준의 SOTA(state of the art) 모델보다 상당한 성능 향상을 얻었고, 단일 프레임 평가에서 21.2%의 top-1 오류와 5.6%의 top-5 오류를 달성했습니다. 이 결과는 2500만개 이하의 파라미터와 단일 추론 당 50억번의 곱셈-덧셈 연산의 계산 비용으로 달성되었습니다. 또한 4개 모델의 앙상블(ensemble)과 다중-크롭 평가(multi-crop evaluation)을 이용하여, 17.3%의 top-1 오류와 3.6%의 top-5 오류를 평가(validation) 데이터셋에서 달성합니다.
사전 훈련된 모델로 ImageNet 데이터셋에서의 단일-크롭 방식으로 오류 비율을 측정한 결과는 아래와 같습니다.

| 모델 구조 | Top-1 오류 | Top-5 오류 |
| --------------- | ----------- | ----------- |
|  inception_v3        | 22.55       | 6.44        |

### 참고문헌

 - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).