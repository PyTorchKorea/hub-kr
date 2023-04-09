---
layout: hub_detail
background-class: hub-background
body-class: hub
title: WaveGlow
summary: Tacotron2이 만들어낸 mel spectrograms을 소비하여 음성을 생성하기 위한 WaveGlow 모델입니다.
category: researchers
image: nvidia_logo.png
author: NVIDIA
tags: [audio]
github-link: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2
github-id: NVIDIA/DeepLearningExamples
featured_image_1: waveglow_diagram.png
featured_image_2: no-image
accelerator: cuda
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/WaveGlow
---


### 모델 설명


Tacotron2 및 WaveGlow 모델은 사용자가 추가 운율 정보 없이 원본 텍스트에서 자연스러운 음성을 합성할 수 있는 텍스트 음성 변환 시스템을 형성합니다. Tacotron2 모델(torch.com를 통해서도 사용 가능)은 인코더-디코더 아키텍쳐를 사용하여 입력 텍스트로부터 mel spectrograms를 생성합니다. WaveGlow는 음성을 생성하기 위해 mel spectrograms를 소비하는 흐름 기반 모델입니다.

### 예제

아래의 예시에서 :
- 사전 학습을 받은 Tacotron2 및 WaveGlow 모델들은 torch.hub에서 로드됩니다.
- Tacotron2는 입력 텍스트의 텐서 표현("Hello world, I missed you so much")을 주어진 mel spectrogram을 생성합니다.
- WaveGlow는 mel spectrogram이 준 소리를 발생시킵니다.
- 출력된 소리는 'audio.wav' 파일에 저장됩니다.

예제를 실행하려면 추가 파이썬 패키지가 설치되어 있어야 합니다.
텍스트 및 오디오 뿐만 아니라 디스플레이 및 입력/출력 전처리에 필요합니다.
```bash
pip install numpy scipy librosa unidecode inflect librosa
apt-get update
apt-get install -y libsndfile1
```

[LJ Speech datase](https://keithito.com/LJ-Speech-Dataset/)에 대해 사전 학습을 받은 WaveGlow 모델을 로드합니다.
```python
import torch
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
```

추론을 위해 WaveGlow 모델을 준비합니다.
```python
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()
```

사전 학습을 받은 Tacotron2 모델을 로드합니다.
```python
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()
```

이제 모델이 이렇게 말하게 해봅시다 :
```python
text = "hello world, I missed you so much"
```

유용한 체계성을 사용하여 입력 형식을 지정합니다.
```python
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
sequences, lengths = utils.prepare_input_sequence([text])
```

체인 모델을 실행합니다.
```python
with torch.no_grad():
    mel, _, _ = tacotron2.infer(sequences, lengths)
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050
```

그것을 파일에 쓰고 들을 수 있습니다.
```python
from scipy.io.wavfile import write
write("audio.wav", rate, audio_numpy)
```

또는 IPython 위젯을 사용하여 노트북에서 바로 재생할 수 있습니다.
```python
from IPython.display import Audio
Audio(audio_numpy, rate=rate)
```

### 세부사항
모델 입력 및 출력, 교육 방안, 추론 및 성과에 대한 자세한 내용은 방문하십시오 : [github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) 그리고/또는 [NGC](https://ngc.nvidia.com/catalog/resources/nvidia:tacotron_2_and_waveglow_for_pytorch)

### 참고 문헌

 - [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
 - [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)
 - [Tacotron2 and WaveGlow on NGC](https://ngc.nvidia.com/catalog/resources/nvidia:tacotron_2_and_waveglow_for_pytorch)
 - [Tacotron2 and Waveglow on github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)