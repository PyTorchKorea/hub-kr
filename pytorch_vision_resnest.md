---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ResNeSt
summary: A new ResNet variant.
category: researchers
image: resnest.jpg
author: Hang Zhang
tags: [vision]
github-link: https://github.com/zhanghang1989/ResNeSt
github-id: zhanghang1989/ResNeSt
featured_image_1: resnest.jpg
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/ResNeSt
---

```python
import torch
# 모델 목록 불러오기
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
# 예시로 ResNeSt-50을 사용하여 사전 훈련된 모델을 불러오기
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
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
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# GPU 사용이 가능한 경우 속도를 위해 입력과 모델을 GPU로 이동
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# ImageNet의 1000개 클래스에 대한 신뢰도 점수를 가진 1000 형태의 Tensor 출력
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

ResNeSt 모델은 [ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf) 논문에서 제안되었습니다.

최근 이미지 분류 모델이 계속 발전하고 있지만 객체 감지 및 의미 분할과 같은 대부분의 다운스트림 애플리케이션(downstream applications)은 간단하게 모듈화된 구조로 인해 여전히 ResNet 변형을 백본 네트워크로 사용합니다. 기능 맵 그룹 전반에 걸쳐 주의를 기울일 수 있는 Split-Attention 블록을 제시합니다. 이러한 Split-Attention 블록을 ResNet 스타일로 쌓아서 ResNeSt라고 하는 새로운 ResNet 변형을 얻습니다. ResNeSt 모델은 유사한 모델 복잡성을 가진 다른 네트워크보다 성능이 우수하며 객체 감지, 인스턴스 분할 및 의미 분할을 포함한 다운스트림 작업을 지원합니다.

|             | crop size | PyTorch |
|-------------|-----------|---------|
| ResNeSt-50  | 224       | 81.03   |
| ResNeSt-101 | 256       | 82.83   |
| ResNeSt-200 | 320       | 83.84   |
| ResNeSt-269 | 416       | 84.54   |

### 참고문헌

 - [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955).
