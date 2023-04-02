---
layout: hub_detail
background-class: hub-background
body-class: hub
title: ResNet50
summary: ResNet50 model trained with mixed precision using Tensor Cores.
category: researchers
image: nvidia_logo.png
author: NVIDIA
tags: [vision]
github-link: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5
github-id: NVIDIA/DeepLearningExamples
featured_image_1: classification.jpg
featured_image_2: no-image
accelerator: cuda
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/ResNet50
---


### 모델 설명

***ResNet50 v1.5***모델은 [original ResNet50 v1 model](https://arxiv.org/abs/1512.03385)의 수정된 버전입니다.

v1과 v1.5의 차이점은 다운샘플링이 필요한 병목 블록에서 v1은 첫 번째 1x1 컨볼루션에서 스트라이드 = 2를 갖는 반면 v1.5는 3x3 컨볼루션에서 스트라이드 = 2를 갖는다는 것입니다.

이러한 차이는 ResNet50 v1.5를 v1보다 조금 더 정확하게 만들지만(\~0.5% top1) 약간의 성능적인 단점(\~5% imgs/sec)이 있습니다.

모델은 [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf)에 설명된 대로 초기화됩니다.

이 모델은 Volta, Turing 및 NVIDIA Ampere GPU 아키텍처의 Tensor 코어를 사용하여 혼합 정밀도(mixed precision)로 학습됩니다. 따라서 연구자들은 혼합 정밀 교육의 이점을 경험하면서 Tensor Core 없이 학습하는 것보다 2배 이상 빠른 결과를 얻을 수 있습니다. 이 모델은 시간이 지남에 따라 일관된 정확성과 성능을 보장하기 위해 각 NGC 월별 컨테이너 릴리스에 대해 테스트됩니다.

ResNet50 v1.5 모델은 TorchScript, ONNX Runtime 또는 TensorRT를 실행 백엔드로 사용하여 [NVIDIA Triton Inference Server](https://github.com/NVIDIA/trtis-inference-server)에서 추론을 위해 배치될 수 있습니다. 자세한 내용은 [NGC](https://ngc.nvidia.com/catalog/resources/nvidia:resnet_for_triton_from_pytorch)를 확인하십시오.

### 예시 사례

아래 예제에서는 사전 훈련된 ***ResNet50 v1.5*** 모델을 사용하여 ***이미지***에 대한 추론을 수행 하고 결과를 제시할 것입니다.

예제를 실행하려면 몇 가지 추가 파이썬 패키지가 설치되어 있어야 합니다. 이는 이미지를 전처리하고 시각화하는 데 필요합니다.
```python
!pip install validators matplotlib
```

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
```

IMAGENET 데이터셋에서 사전 훈련된 모델을 로드합니다.
```python
resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

resnet50.eval().to(device)
```

샘플 입력 데이터를 준비합니다.
```python
uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)
```

추론을 실행합니다. `pick_n_best(predictions=output, n=topN)` helper 함수를 사용하여 모델에 따라 가장 가능성이 높은 가설을 N개 선택합니다.
```python
with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(batch), dim=1)
    
results = utils.pick_n_best(predictions=output, n=5)
```

결과를 표시합니다.
```python
for uri, result in zip(uris, results):
    img = Image.open(requests.get(uri, stream=True).raw)
    img.thumbnail((256,256), Image.ANTIALIAS)
    plt.imshow(img)
    plt.show()
    print(result)

```

### 세부사항
모델 입력 및 출력, 학습 방법, 추론 및 성능 등에 대한 더 자세한 정보는 [github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5) 및 and/or [NGC](https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch)에서 볼 수 있습니다.


### 참고문헌

 - [Original ResNet50 v1 paper](https://arxiv.org/abs/1512.03385)
 - [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf)
 - [model on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5)
 - [model on NGC](https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch)
 - [pretrained model on NGC](https://ngc.nvidia.com/catalog/models/nvidia:resnet50_pyt_amp)
 
