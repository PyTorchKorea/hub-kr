---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: SlowFast
summary: Kinetics 400 데이터섯에서 사전 학습된 SlowFast 네트워크
image: slowfast.png 
author: FAIR PyTorchVideo
tags: [vision]
github-link: https://github.com/facebookresearch/pytorchvideo
github-id: facebookresearch/pytorchvideo
featured_image_1: no-image 
featured_image_2: no-image
accelerator: “cuda-optional” 
demo-model-link: https://huggingface.co/spaces/pytorch/SlowFast
---

### 사용 예시

#### 불러오기

모델 불러오기: 

```python
import torch
# `slowfast_r50` 모델 선택
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
```

나머지 함수들 불러오기:

```python
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
```

#### 셋업

모델을 평가 모드로 설정하고 원하는 디바이스 방식을 선택합니다.

```python 
# GPU 또는 CPU 방식을 설정합니다.
device = "cpu"
model = model.eval()
model = model.to(device)
```

토치 허브 모델이 훈련된 Kinetics 400 데이터셋을 위한 id-레이블 매핑 정보를 다운로드합니다. 이는 예측된 클래스 id에 카테고리 레이블 이름을 붙이는 데 사용됩니다.

```python
json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
json_filename = "kinetics_classnames.json"
try: urllib.URLopener().retrieve(json_url, json_filename)
except: urllib.request.urlretrieve(json_url, json_filename)
```

```python
with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# id-레이블 이름 매핑 만들기
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")
```

#### 입력 형태에 대한 정의

```python
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3

class PackPathway(torch.nn.Module):
    """
    영상 프레임을 텐서 리스트로 바꾸기 위한 변환.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

# 입력 클립의 길이는 모델에 따라 달라집니다.
clip_duration = (num_frames * sampling_rate)/frames_per_second
```

#### 추론 실행

예제 영상을 다운로드합니다.

```python
url_link = "https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4"
video_path = 'archery.mp4'
try: urllib.URLopener().retrieve(url_link, video_path)
except: urllib.request.urlretrieve(url_link, video_path)
```

영상을 불러오고 모델에 필요한 입력 형식으로 변환합니다.

```python
# 시작 및 종료 구간을 지정하여 불러올 클립의 길이를 선택합니다.
# start_sec는 영상에서 행동이 시작되는 위치와 일치해야 합니다.
start_sec = 0
end_sec = start_sec + clip_duration

# EncodedVideo helper 클래스를 초기화하고 영상을 불러옵니다.
video = EncodedVideo.from_path(video_path)

# 원하는 클립을 불러옵니다.
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

# 영상 입력을 정규화하기 위한 변환(transform 함수)을 적용합니다.
video_data = transform(video_data)

# 입력을 원하는 디바이스로 이동합니다.
inputs = video_data["video"]
inputs = [i.to(device)[None, ...] for i in inputs]
```

#### 예측값 구하기

```python
# 모델을 통해 입력 클립을 전달합니다.
preds = model(inputs)

# 예측된 클래스를 가져옵니다.
post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices[0]

# 예측된 클래스를 레이블 이름에 매핑합니다.
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))
```

### 모델 설명
SlowFast 모델 아키텍처는 Kinetics 데이터셋의 8x8 설정을 사용하여 사전 훈련된 가중치가 있는 [1]을 기반으로 합니다.

| arch | depth | frame length x sample rate | top 1 | top 5 | Flops (G) | Params (M) |
| --------------- | ----------- | ----------- | ----------- | ----------- | ----------- |  ----------- | ----------- |
| SlowFast | R50   | 8x8                        | 76.94 | 92.69 | 65.71     | 34.57      |
| SlowFast | R101  | 8x8                        | 77.90 | 93.27 | 127.20    | 62.83      |


### 참고문헌
[1] Christoph Feichtenhofer et al, "SlowFast Networks for Video Recognition"
https://arxiv.org/pdf/1812.03982.pdf
