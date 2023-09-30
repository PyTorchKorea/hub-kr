---
layout: hub_detail
background-class: hub-background
body-class: hub
title: FCN
summary: ResNet-50 및 ResNet-101 백본을 사용하는 완전 컨볼루션 네트워크 모델
category: researchers
image: fcn2.png
author: Pytorch Team
tags: [vision, scriptable]
github-link: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py
github-id: pytorch/vision
featured_image_1: deeplab1.png
featured_image_2: fcn2.png
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/FCN
---

```python
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
# or
# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet101', pretrained=True)
model.eval()
```

모든 사전 훈련된 모델은 동일한 방식으로 정규화된 입력 이미지, 즉 `N`이 이미지 수이고, `H`와 `W`는 최소 `224`픽셀인 `(N, 3, H, W)`형태의 3채널 RGB 이미지의 미니 배치를 요구합니다. 
이미지를 `[0, 1]` 범위로 로드한 다음 `mean = [0.485, 0.456, 0.406]` 및 `std = [0.229, 0.224, 0.225]`를 사용하여 정규화해야 합니다.
모델은 입력 텐서와 높이와 너비는 같지만 클래스가 21개인 텐서를 가진 `OrderedDict`를 반환합니다. `output['out']`에는 시멘틱 마스크가 포함되며 `output['aux']`에는 픽셀당 보조 손실 값이 포함됩니다. 추론 모드에서는 `output['aux']`이 유용하지 않습니다.
그래서 `output['out']`의 크기는 `(N, 21, H, W)`입니다. 추가 설명서는 [여기](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)에서 찾을 수 있습니다.


```python
# 파이토치 웹사이트에서 예제 이미지 다운로드
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 실행 예시 (torchvision 필요)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
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
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)
```

여기서의 출력 형태는 `(21, H, W)`이며, 각 위치에는 각 클래스의 예측에 해당하는 정규화되지 않은 확률이 있습니다. 각 클래스의 최대 예측을 가져온 다음 이를 다운스트림 작업에 사용하려면 `output_propertions = output.slmax(0)`를 수행합니다. 다음은 각 클래스에 할당된 각 색상과 함께 예측을 표시하는 작은 토막글 입니다(왼쪽의 시각화 이미지 참조).

```python
# 각 클래스에 대한 색상을 선택하여 색상 팔레트를 만듭니다.
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# 각 색상의 21개 클래스의 시멘틱 세그멘테이션 예측을 그림으로 표시합니다.
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
plt.imshow(r)
# plt.show()
```

### 모델 설명

FCN-ResNet은 ResNet-50 또는 ResNet-101 백본을 사용하여 완전 컨볼루션 네트워크 모델로 구성됩니다. 사전 훈련된 모델은 Pascal VOC 데이터 세트에 존재하는 20개 범주에 대한 COCO 2017의 하위 집합에 대해 훈련 되었습니다.

COCO val 2017 데이터셋에서 평가된 사전 훈련된 모델의 정확성은 아래에 나열되어 있습니다.

| Model structure |   Mean IOU  | Global Pixelwise Accuracy |
| --------------- | ----------- | --------------------------|
|  fcn_resnet50   |   60.5      |   91.4                    |
|  fcn_resnet101  |   63.7      |   91.9                    |

### Resources

 - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211)
