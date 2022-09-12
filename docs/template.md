---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
<!-- pytorch_vision_fcn_resnet101.md를 참고하여 아래 라인을 변경해주시고 작성해주시고, 이 줄은 지워주세요. -->
title: <필수 : 짦은 모델 이름>
summary: <필수 : 모델을 설명하는 1~2 문장>
image: <필수 : 모델을 나타내는 대표 이미지>
author: <필수 : 작성자 이름>
tags: <필수 : [tag1, tag2, ...] 허용되는 태그는 vision, nlp, generative, audio, scriptable 입니다.>
github-link: <필수 : 깃허브 링크>
github-id: <필수 : 저장소 최상단에 표현되는 아이디>
featured_image_1: <선택 : 모델을 설명하는 이미지1>
featured_image_2: <선택 : 모델을 설명하는 이미지2>
accelerator: <선택 : 현재 지원되는 옵션 : "cuda", "cuda-optional">
---
<!-- 필수 : torch.hub 를 통해 실행할 수 있는 예제 스크립트를 넣어주세요. -->
```python
import torch
torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
```
<!-- 모델을 설명하는 코드를 넣어주세요. 25줄 이하가 적합합니다. -->

<!-- 필수 : 모델 설명을 넣어주세요, md format 으로 작성가능합니다. -->
### Model Description


<!-- 선택 : 참조 논문의 링크를 넣어주세요. -->
### References
