---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Progressive Growing of GANs (PGAN)
summary: High-quality image generation of fashion, celebrity faces
category: researchers
image: pganlogo.png
author: FAIR HDGAN
tags: [vision, generative]
github-link: https://github.com/facebookresearch/pytorch_GAN_zoo/blob/master/models/progressive_gan.py
github-id: facebookresearch/pytorch_GAN_zoo
featured_image_1: pgan_mix.jpg
featured_image_2: pgan_celebaHQ.jpg
accelerator: cuda-optional
order: 10
demo-model-link: https://colab.research.google.com/drive/19NTYFNUT9js78UZ0g_3IsnnUW9AsR0fD?usp=sharing
---


```python
import torch
use_gpu = True if torch.cuda.is_available() else False

# 유명인들의 고해상도 얼굴사진들로 만든 "celebA" 데이터셋으로 훈련시켰습니ㅏ
# 이 모델은 512x512 픽셀 크기의 이미지를 출력하게 됩니다
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=use_gpu)
# 이 모델은 256x5126 픽셀 크기의 이미지를 출력하게 됩니다
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-256',
#                        pretrained=True, useGPU=use_gpu)
```

모델의 입력값으로 들어가는 노이즈(noise) 벡터는 `(N, 512)` 크기를 갖는데, 이때 `N` 은 생성하고싶은 데이터의 수입니다.
이 노이즈 벡터들은 `.buildNoiseData`를 이용해서 만들 수 있습니다.
모델은 `.test` 함수를 갖는데, 이 함수는 노이즈를 받아서 이미지를 생성하는 함수입니다.

```python
num_images = 4
noise, _ = model.buildNoiseData(num_images)
with torch.no_grad():
    generated_images = model.test(noise)

# 생성한 이미지들을 torchvision과 matplotlib을 이용해서 시각화를 해봅시다
import matplotlib.pyplot as plt
import torchvision
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
# plt.show()
```

결과를 확인해보면 , 오른쪽의 이미지와 비슷한 사진을 확인 할 수 있을겁니다

만약 자신만의 Progressive GAN 이나 여타 GAN모델들을 밑바닥부터 구현해보고 싶다면, [PyTorch GAN Zoo](https://github.com/facebookresearch/pytorch_GAN_zoo)을 확인해보시기 바랍니다.

### 모멜 설명

컴퓨터 비전(computer vision)에서 생성자(generative) 모델이란, 주어진 입력값을 토대로 새로운 이미지를 만들어내는 신경망을 뜻합니다.
지금 우리가 다루는 것은 생성자 신경망의 특정 모델로서: 무작위의 벡터를 사실적인 이미지로 변환시켜주는 GANs (Generative Adversarial Networks) 모델 입니다.

GAN의 점진적 학습(Progressive Growing of GANs)은 Karras와 그 외[1]가 2017년 발표한 방법론으로, 고해상도의 이미지를 생성할 수 있습니다. 이를 위해 생성자 모델은 부분적으로 차근차근 훈련을 하게됩니다. 처음에는 가장 낮은 해상도를 학습하도록 하고, 어느정도 모델이 수렴하게 되면, 새로운 계층(layer)이 모델에 더해져, 기존 출력 이미지 크기의 2배로 커진 출력값으로 이어서 훈련하는 방식입니다. 이 과정은 원하는 해상도에 도달할때까지 반복됩니다.

### 준비물

- 현재는 Python 3에서만 지원이 됩니다

### 참고

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
