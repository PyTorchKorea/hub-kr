---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ProxylessNAS
summary: Proxylessly specialize CNN architectures for different hardware platforms.
category: researchers
image: proxylessnas.png
author: MIT Han Lab
tags: [vision]
github-link: https://github.com/mit-han-lab/ProxylessNAS
github-id: mit-han-lab/ProxylessNAS
featured_image_1: proxylessnas.png
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/ProxylessNAS
---

```python
import torch
target_platform = "proxyless_cpu"
# proxyless_gpu, proxyless_mobile, proxyless_mobile14도 사용할 수 있습니다.
model = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=True)
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
input_batch = input_tensor.unsqueeze(0) # 모델에서 요구하는 미니배치 생성

# GPU 사용이 가능한 경우 속도를 위해 입력과 모델을 GPU로 이동
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# ImageNet 1000개 클래스에 대한 신뢰도 점수를 가진 1000 형태의 Tensor 출력
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

ProxylessNAS 모델은 [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332) 논문에서 제안되었습니다.

일반적으로, 사람들은 *모든 하드웨어 플랫폼*에 대해 *하나의 효율적인 모델*을 설계하는 경향이 있습니다. 하지만 하드웨어마다 특성이 다릅니다. 예를 들어 CPU는 더 높은 주파수를 가지지만 GPU는 병렬화에 더 뛰어납니다. 따라서 모델을 일반화하기보다는 하드웨어 플랫폼에 맞게 CNN 아키텍처를 **전문화**해야 합니다. 아래에서 볼 수 있듯이, 전문화는 세 가지 플랫폼 모두에서 상당한 성능 향상을 제공합니다.

| Model structure |  GPU Latency | CPU Latency | Mobile Latency
| --------------- | ----------- | ----------- | ----------- |
|  proxylessnas_gpu     |  **5.1ms**   | 204.9ms | 124ms |
|  proxylessnas_cpu     |  7.4ms   | **138.7ms** | 116ms |
|  proxylessnas_mobile  |  7.2ms   | 164.1ms | **78ms**  |

사전 훈련된 모델에 해당하는 Top-1 정확도는 아래에 나열되어 있습니다.

| Model structure | Top-1 error |
| --------------- | ----------- |
|  proxylessnas_cpu     |  24.7 |
|  proxylessnas_gpu     |  24.9   |
|  proxylessnas_mobile  |  25.4   |
|  proxylessnas_mobile_14  |  23.3   |

### 참고문헌

 - [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332).
