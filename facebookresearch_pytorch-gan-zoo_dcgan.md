---
layout: hub_detail
background-class: hub-background
body-class: hub
title: DCGAN on FashionGen
summary: 64x64 이미지 생성을 위한 기본 이미지 생성 모델
category: researchers
image: dcgan_fashionGen.jpg
author: FAIR HDGAN
tags: [vision, generative]
github-link: https://github.com/facebookresearch/pytorch_GAN_zoo/blob/master/models/DCGAN.py
github-id: facebookresearch/pytorch_GAN_zoo
featured_image_1: dcgan_fashionGen.jpg
featured_image_2: no-image
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/DCGAN_on_fashiongen
order: 10
---

```python
import torch
use_gpu = True if torch.cuda.is_available() else False

model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=use_gpu)
```

모델에 입력하는 잡음(noise) 벡터의 크기는 `(N, 120)` 이며 여기서 `N`은 생성하고자 하는 이미지의 개수입니다. 데이터 생성은 `.buildNoiseData` 함수를 사용하여 데이터를 생성할 수(구성할 수) 있습니다. 모델의 `.test` 함수를 사용하면 노이즈 벡터를 입력받아 이미지를 생성합니다.

```python
num_images = 64
noise, _ = model.buildNoiseData(num_images)
with torch.no_grad():
    generated_images = model.test(noise)

# let's plot these images using torchvision and matplotlib
import matplotlib.pyplot as plt
import torchvision
plt.imshow(torchvision.utils.make_grid(generated_images).permute(1, 2, 0).cpu().numpy())
# plt.show()
```

왼쪽에 있는 이미지와 유사하다는것을 볼 수 있습니다.

만약 자기만의 DCGAN과 다른 GAN을 처음부터 학습시키고 싶다면, [PyTorch GAN Zoo](https://github.com/facebookresearch/pytorch_GAN_zoo) 참고하시기 바랍니다.

### 모델 설명

컴퓨터 비전에서 생성 모델은 주어진 입력값으로부터 이미지를 생성하도록 훈련된 네트워크(networks)입니다. 본 예제에서는 특정 종류의 이미지만 생성되도록 고려한 생성 네트워크: 무작위 벡터를 실제 이미지 생성과 연결하는 방법을 배우는 GANs (Generative Adversarial Networks) 입니다.

DCGAN은 2015년 Radford 등이 설계한 모델 구조입니다. 상세한 내용은 [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) 논문에서 확인할 수 있습니다. 모델은 GAN 구조이며 저해상도 이미지 (최대 64x64) 생성에 매우 간편하고 효율적입니다.


### 요구 사항

- 현재는 오직 Python 3 에서만 지원됩니다.

### 참고문헌

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
