---
layout: hub_detail
background-class: hub-background
body-class: hub
title: GhostNet
summary: Efficient networks by generating more features from cheap operations
category: researchers
image: ghostnet.png
author: Huawei Noah's Ark Lab
tags: [vision, scriptable]
github-link: https://github.com/huawei-noah/ghostnet
github-id: huawei-noah/ghostnet
featured_image_1: ghostnet.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/GhostNet
---

```python
import torch
model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
model.eval()
```

모든 사전 학습된 모델들은 입력 이미지가 동일한 방식으로 정규화 되는 것을 요구합니다. 
다시 말해 `H`와 `W`가 적어도 `224`이고, `(3 x H x W)`의 shape를 가지는 3채널 RGB 이미지들의 미니배치를 말합니다.
이 이미지들은 `[0, 1]`의 범위로 로드되어야 하고, `mean = [0.485, 0.456, 0.406]`
과 `std = [0.229, 0.224, 0.225]`를 사용하여 정규화되어야 합니다.

여기서부터는 예시 코드 입니다.

```python
# pytorch 웹사이트에서 예시 이미지를 다운로드 합니다.
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
```

```python
# 실행 예시 코드 (torchvision 필요합니다)
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
input_batch = input_tensor.unsqueeze(0) # 모델에 맞추어 미니배치를 생성 합니다.

# 연산속도를 위해 input과 모델을 GPU에 로드 합니다
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# ImageNet 1000개의 클래스의 신뢰점수를 포함하는 (1000,) 의 텐서를 return 합니다.
print(output[0])
# output은 정규화되지 않은 신뢰 점수로 얻어집니다. 확률을 얻기 위해 소프트맥스를 사용할 수 있습니다.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
```

```
# ImageNet의 라벨을 다운로드 합니다.
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

```
# 카테고리를 읽어옵니다.
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# 이미지 마다 확률값이 가장 높은 범주 출력 합니다.
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

### 모델 설명

고스트넷 아키텍처는 다양한 특징 맵을 효율적인 연산으로 생성하는 고스트 모듈 구조로 이루어집니다. 
합성곱 신경망에서의 학습 과정에서 추론에 중요한 중복되는 고유 특징맵(고스트 맵)들이 다수 생성되는 현상에 기반하여 설계 되었습니다. 고스트넷에서는 더 효율적인 연산으로 고스트 맵들을 생성합니다.
벤치마크에서 수행된 실험을 통해 속도와 정확도의 상충 관계에 관한 고스트넷의 우수성을 보여줍니다.

사전 학습된 모델을 사용한 ImageNet 데이터셋에 따른 정확도는 아래에 나열되어 있습니다.

| Model structure | FLOPs       | Top-1 acc   | Top-5 acc   |
| --------------- | ----------- | ----------- | ----------- |
|  GhostNet 1.0x  | 142M        | 73.98       | 91.46       |


### 참고

다음 [링크](https://arxiv.org/abs/1911.11907)에서 논문의 전체적인 내용에 대하여 읽을 수 있습니다.

>@inproceedings{han2019ghostnet,
>    title={GhostNet: More Features from Cheap Operations},
>    author={Kai Han and Yunhe Wang and Qi Tian and Jianyuan Guo and Chunjing Xu and Chang Xu},
>    booktitle={CVPR},
>    year={2020},
>}

