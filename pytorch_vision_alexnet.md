---
layout: hub_detail
background-class: hub-background
body-class: hub
title: AlexNet
summary: 2012년 'ImageNet' 우승자는 상위 5위 안에 드는 15.3%의 오류를 달성했는데, 이는 2위보다 10.8% 이상 낮은 수치입니다.
category: researchers
image: alexnet2.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
github-id: pytorch/vision
featured_image_1: alexnet1.png
featured_image_2: alexnet2.png
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/AlexNet
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()
```

모든 사전 학습된 모델들은 입력 이미지가 동일한 방식으로 정규화될 것으로 요구됩니다.
즉, 미니 배치의 3-채널 RGB 이미지들은 `(3 x H x W)`의 형태를 가지며, `H`와 `W`는 최소 `224` 이상이어야 합니다.
각 이미지들은 `[0, 1]`의 범위 내에서 불러와야 하며, `mean = [0.485, 0.456, 0.406]` 과 `std = [0.229, 0.224, 0.225]`을 이용해 정규화되어야 합니다.

다음은 실행 예제 입니다.

```python
# pytorch 웹사이트에서 예제 다운로드
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 요구되는 미니배치 생성

# 사용 가능한 경우 속도를 위해 인풋(input)과 모델(model)을 GPU로 이동
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# 아웃풋(output)은 1000개의 Tensor 형태이며, 이는 Imagenet 데이터 셋의 1000개 클래스에 대한 신뢰도 점수를 나타내는 결과
print(output[0])
# 아웃풋의 결과는 정규화되지 않는 결과를 가집니다. 개연성을 얻기 위해서, 소프트맥스(softmax)를 실행할 수 있습니다.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet의 라벨을 다운로드 합니다
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리를 읽어옵니다
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 이미지 마다 확률값이 가장 높은 범주 출력 합니다
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

알렉스넷(AlexNet)은 2012년 9월 30일에 열린 'ImageNet' 대규모 시각 인식 챌린지에 참여했습니다. 이 네트워크는 상위 5위 안에 드는 오류를 15.3%로 2위보다 10.8% 이상 낮은 수치였습니다.
원본 논문의 주요 결과는 모델의 깊이가 높은 성능을 위해 필수적이어서 계산 비용이 많이 들었지만 훈련 중 그래픽 처리 장치(GPUs)의 활용으로 실현 되었다는 것입니다.

The 1-crop error rates on the imagenet dataset with the pretrained model are listed below.
사전 훈련된 모델이 있는 이미지넷 데이터 세트의 '1-crop' 오류율은 다음과 같습니다.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  alexnet        | 43.45       | 20.91       |

### 참고

1. [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997).
