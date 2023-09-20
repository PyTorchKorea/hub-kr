---
layout: hub_detail
background-class: hub-background
body-class: hub
title: AlexNet
summary: 2012년 ImageNet 우승자는 15.3%의  top-5 에러율을 달성하여 준우승자보다 10.8%P 이상 낮았습니다.
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

모든 사전 훈련된 모델은 동일한 방식으로 정규화된 입력 이미지, 즉 N이 이미지 수이고, H와 W는 최소 224픽셀인 (N, 3, H, W)형태의 3채널 RGB 이미지의 미니 배치를 요구합니다. 이미지를 [0, 1] 범위로 로드한 다음 mean = [0.485, 0.456, 0.406] 및 std = [0.229, 0.224, 0.225]를 사용하여 정규화해야 합니다.

이미지는 `[0, 1]`의 범위에서 로드되어야 하고 `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` 으로 정규화해야합니다.

다음은 실행 예제입니다.

```python
# PyTorch 웹사이트에서 예제 이미지 다운로드
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 요구하는 형식인 미니 배치 생성

# 빠른 실행을 위해 GPU 사용 가능 시 모델과 입력값을 GPU를 사용하도록 설정
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Imagenet 1000개 클래스의 신뢰 점수를 나타내는 텐서

print(output[0])

# 결과는 비정규화된 점수입니다. softmax으로 돌리면 확률값을 얻을 수 있습니다.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```python
# ImageNet 레이블 다운로드
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```python
# 카테고리 읽기
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 이미지별 확률값 상위 카테고리 출력
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

AlexNet은 2012년도 ImageNet Large Scale Visual Recognition Challenge (ILSVRC)에 참여한 모델입니다. 이 네트워크는 15.3%의 top-5 에러율을 달성했고, 이는 2위보다 10.8%P 낮은 수치입니다. 원 논문의 주요 결론은 높은 성능을 위해 모델의 깊이가 필수적이라는 것이었습니다. 이는 계산 비용이 많이 들지만, 학습 과정에서 GPU의 사용으로 가능해졌습니다.

사전 훈련된 모델이 있는 ImageNet 데이터셋의 1-crop 에러율은 다음 표와 같습니다.

| 모델 구조 | Top-1 에러 | Top-5 에러 |
| --------------- | ----------- | ----------- |
|  alexnet        | 43.45       | 20.91       | -->

### 참고문헌

1. [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997).
