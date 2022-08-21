---
layout: hub_detail
background-class: hub-background
body-class: hub
category: researchers
title: YOLOv5
summary: YOLOv5 in PyTorch > ONNX > CoreML > TFLite
image: ultralytics_yolov5_img0.jpg
author: Ultralytics
tags: [vision, scriptable]
github-link: https://github.com/ultralytics/yolov5
github-id: ultralytics/yolov5
featured_image_1: ultralytics_yolov5_img1.jpg
featured_image_2: ultralytics_yolov5_img2.png
accelerator: cuda-optional
demo-model-link: https://huggingface.co/spaces/pytorch/YOLOv5
---

## Before You Start

**Python>=3.8**과 **PyTorch>=1.7** 환경을 갖춘 상태에서 시작해주세요. PyTorch를 설치해야 한다면 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 를 참고하세요. YOLOv5 dependency를 설치하려면:

```bash
pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # 필요한 모듈 설치
```


## Model Description

<img width="800" alt="YOLOv5 Model Comparison" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png">
&nbsp;

[YOLOv5](https://ultralytics.com/yolov5) 🚀는 compound-scaling을 사용하고 COCO dataset으로 학습한 Object detection 모델들 중 하나이며, Test Time Augmentation (TTA), 모델 앙상블(model ensembling), 하이퍼파라미터 평가(hyperparameter evolution), 그리고 ONNX, CoreML과 TFLite로 변환(export)을 간단하게 해주는 기능이 포함되어 있습니다.

|모델 |크기<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |속도<br><sup>V100 (ms) | |파라미터 수<br><sup>(M) |FLOPS<br><sup>640 (B)
|---   |---  |---        |---         |---             |---                |---|---              |---
|[YOLOv5s6](https://github.com/ultralytics/yolov5/releases)   |1280 |43.3     |43.3     |61.9     |**4.3** | |12.7  |17.4
|[YOLOv5m6](https://github.com/ultralytics/yolov5/releases)   |1280 |50.5     |50.5     |68.7     |8.4     | |35.9  |52.4
|[YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |1280 |53.4     |53.4     |71.1     |12.3    | |77.2  |117.7
|[YOLOv5x6](https://github.com/ultralytics/yolov5/releases)   |1280 |**54.4** |**54.4** |**72.0** |22.4    | |141.8 |222.9
|[YOLOv5x6](https://github.com/ultralytics/yolov5/releases) TTA |1280 |**55.0** |**55.0** |**72.0** |70.8 | |-  |-

<details>
  <summary>표에 대한 설명 (확장하려면 클릭)</summary>

  * AP<sup>test</sup> 는 COCO [test-dev2017](http://cocodataset.org/#upload) 서버에서 평가한 결과이고, 나머지 AP 결과들은 val2017 데이터셋에 대한 결과를 의미합니다.
  * 달리 명시되지 않은 한, AP 값들은 단일 모델, 단일 규모(scale)로부터 얻은 값입니다. **mAP 재현**은 `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65` 을 실행하면 가능합니다.
  * 속도<sub>GPU</sub>는 GCP의 [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 인스턴스를 사용하여 총 5000장의 COCO val2017 이미지 각각에 대한 추론 속도를 평균 내어 구하였으며, FP16 추론과 후처리, NMS 시간이 포함되어 있습니다. **속도 재현**은 `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45` 을 실행하면 가능합니다.
  * 모든 체크포인트는 기본 세팅과 하이퍼파라미터(자동증강 없음)로 300 에폭까지 학습한 결과입니다.
  * Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) 은 반사(reflection)와 규모(scale) 증강을 포함합니다. **TTA 재현**은 `python test.py --data coco.yaml --img 1536 --iou 0.7 --augment` 을 실행하면 가능합니다.

</details>

<p align="left"><img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/model_plot.png"></p>

<details>
  <summary>그림에 대한 설명 (확장하려면 클릭)</summary>

  * GPU 속도는 V100 GPU에서 배치 크기를 32로 설정한 환경에서 총 5000장의 COCO val2017 이미지 각각에 대한 end-to-end 연산 시간을 평균 내어 구하였으며, 속도 측정 구간은 이미지 전처리와 Pytorch FP16 추론, 후처리와 NMS 과정을 포함합니다.
  * EfficientDet 데이터는 [google/automl](https://github.com/google/automl) 의 배치 크기 8인 모델에 대한 데이터입니다.
  * **재현** 하려면 `python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt` 을 실행하면 가능합니다.

</details>

## Load From PyTorch Hub


이 예제에서는 사전 훈련된(pretrained) **YOLOv5s** 모델을 불러와 이미지에 대해 추론을 진행합니다. YOLOv5s는 **URL**, **파일 이름**, **PIL**, **OpenCV**, **Numpy**와 **PyTorch** 형식의 입력을 받고, **torch**, **pandas**, **JSON** 출력 형태로 탐지 결과를 반환합니다. 자세한 정보는 [YOLOv5 파이토치 허브 튜토리얼](https://github.com/ultralytics/yolov5/issues/36) 을 참고하세요.


```python
import torch

# 모델
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 이미지
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# 추론
results = model(imgs)

# 결과
results.print()
results.save()  # 혹은 .show()

results.xyxy[0]  # img1에 대한 예측 (tensor)
results.pandas().xyxy[0]  # img1에 대한 예측 (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
```


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## Contact


**이슈가 생기면 즉시 https://github.com/ultralytics/yolov5 로 알려주세요.** 비즈니스 상의 문의나 전문적인 지원 요청은 [https://ultralytics.com](https://ultralytics.com) 을 방문하거나 Glenn Jocher의 이메일인 [glenn.jocher@ultralytics.com](mailto:glenn.jocher@ultralytics.com) 으로 연락 주세요.


&nbsp;
