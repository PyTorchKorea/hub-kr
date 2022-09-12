---
layout: hub_detail
background-class: hub-background
body-class: hub
title: MobileNet v2
summary: 잔차 블록에 기반한 속도와 메모리에 최적화된 효율적인 네트워크
category: researchers
image: mobilenet_v2_1.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenet.py
github-id: pytorch/vision
featured_image_1: mobilenet_v2_1.png
featured_image_2: mobilenet_v2_2.png
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/MobileNet_v2
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
```

모든 사전 훈련된 모델들은 동일한 방식으로 정규화된 이미지를 입력으로 사용합니다.
즉, 미니 배치의 3-채널 RGB 이미지들은 `(3 x H x W)`의 형태를 가지며, 해당 `H`와 `W`는 최소 `224` 이상이어야 합니다.
각 이미지는 `[0, 1]`의 범위 내에서 불러와야 하며, `mean = [0.485, 0.456, 0.406]` 과 `std = [0.229, 0.224, 0.225]`을 이용해 정규화되어야 합니다.

다음은 실행 예제 입니다.

```python
# pytorch 웹사이트에서 예제 이미지 다운로드
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 실행 예제 (torchvision 필요)
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 상정하는 미니배치 생성

# 사용 가능한 경우 속도를 위해 입력 데이터와 모델을 GPU로 이동
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# output은 1000개의 Tensor 형태이며, 이는 Imagenet 데이터 셋의 1000개 클래스에 대한 신뢰도 점수를 나타내는 결과
print(output[0])
# output 결과는 정규화되지 않은 결과. 확률을 얻기 위해선 softmax를 거쳐야 함.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet 라벨 다운로드
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리 읽어들이기
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 이미지 별 상위 카테고리 표시
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

MobileNet v2 구조는 잔차 블록의 입력 및 출력이 얇은 병목 계층 형태인 반전된 잔차 구조를 기반으로 합니다. 반전된 잔차 구조는 입력단에서 확장된 표현을 사용하는 기존의 잔차 모델과 반대되는 구조입니다. MobileNet v2는 경량화된 depthwise 합성곱을 사용하여 중간 확장 계층의 특징들을 필터링합니다. 또한, 표현력 유지를 위해 좁은 계층의 비선형성은 제거되었습니다.

| 모델 구조 | Top-1 오류 | Top-5 오류 |
| --------------- | ----------- | ----------- |
|  mobilenet_v2       | 28.12       | 9.71       |


### 참고문헌

 - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
