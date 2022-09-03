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
demo-model-link: https://huggingface.co/spaces/pytorch/PGAN
---


```python
import torch
use_gpu = True if torch.cuda.is_available() else False

# 이 모델은 유명인들의 고해상도 얼굴 데이터셋 "celebA"로 학습되었습니다
# 아래 모델의 출력은 512 x 512 픽셀의 이미지입니다
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=use_gpu)
# 아래 모델의 출력은 256 x 256 픽셀의 이미지입니다
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-256',
#                        pretrained=True, useGPU=use_gpu)
```

모델의 입력값으로는 `(N, 512)`크기의 노이즈(noise) 벡터입니다. `N`은 생성하고자 하는 이미지의 개수를 뜻합니다.
이 노이즈 벡터들은 함수 `.buildNoiseData`를 통하여 생성 할 수 있습니다.
이 모델은 노이즈 벡터를 받아서 이미지를 생성하는 `.test` 함수를 가지고 있습니다.

```python
num_images = 4
noise, _ = model.buildNoiseData(num_images)
with torch.no_grad():
    generated_images = model.test(noise)

# torchvision과 matplotlib를 이용하여 생성한 이미지들을 시각화 해봅시다.
import matplotlib.pyplot as plt
import torchvision
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
# plt.show()
```

왼쪽과 비슷한 이미지를 결과물로 확인할 수 있습니다.

만약 자신만의 Progressive GAN 이나 다른 GAN 모델들을 직접 학습해 보고 싶다면 [PyTorch GAN Zoo](https://github.com/facebookresearch/pytorch_GAN_zoo)를 참고해 보시기 바랍니다.

### 모델 설명

컴퓨터 비전(Computer Vision)분야에서 생성 모델은 주어진 입력값으로 부터 이미지를 생성해 내도록 학습된 신경망입니다. 현재 다루는 모델은 생성 모델의 특정한 종류로서 무작위의 벡터에서 사실적인 이미지를 생성하는 법을 학습하는 GAN 모델입니다.

GAN의 점진적인 증가(Progressive Growing of GANs)는 Karras와 그 외[1]가 2017년에 발표한 고해상도의 이미지 생성을 위한 방법론 입니다. 이를 위하여 생성 모델은 여러 단계로 나뉘어서 학습됩니다. 제일 먼저 모델은 아주 낮은 해상도의 이미지를 생성하도록 학습이 되고, 어느정도 모델이 수렴하면 새로운 계층이 모델에 더해지고 출력 해상도는 2배가 됩니다. 이 과정을 원하는 해상도에 도달 할 때 까지 반복합니다.

### 요구사항

- 현재는 Python3 에서만 지원합니다

### References

- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
