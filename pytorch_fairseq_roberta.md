---
layout: hub_detail
background-class: hub-background
body-class: hub
title: RoBERTa
summary: BERT를 강력하게 최적화하는 사전 학습 접근법, RoBERTa
category: researchers
image: fairseq_logo.png
author: Facebook AI (fairseq Team)
tags: [nlp]
github-link: https://github.com/pytorch/fairseq/
github-id: pytorch/fairseq
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/RoBERTa
---


### 모델 설명

Bidirectional Encoder Representations from Transformers, [BERT][1]는 텍스트에서 의도적으로 숨겨진 부분을 예측하는 뛰어난 자기지도 사전 학습(self-supervised pretraining) 기술입니다. 특히 BERT가 학습한 표현은 다운스트림 태스크(downstream tasks)에 잘 일반화되는 것으로 나타났으며, BERT가 처음 공개된 2018년에 수많은 자연어처리 벤치마크 데이터셋에 대해 가장 좋은 성능을 기록했습니다.

[RoBERTa][2]는 BERT의 언어 마스킹 전략(language masking strategy)에 기반하지만 몇 가지 차이점이 존재합니다. 다음 문장 사전 학습(next-sentence pretraining objective)을 제거하고 훨씬 더 큰 미니 배치와 학습 속도로 훈련하는 등 주요 하이퍼파라미터를 수정합니다. 또한 RoBERTa는 더 오랜 시간 동안 BERT보다 훨씬 많은 데이터에 대해 학습되었습니다. 이를 통해 RoBERTa의 표현은 BERT보다 다운스트림 태스크에 더 잘 일반화될 수 있습니다.


### 요구 사항

추가적인 Python 의존성이 필요합니다.

```bash
pip install regex requests hydra-core omegaconf
```


### 예시

##### RoBERTa 불러오기
```python
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # 드롭아웃 비활성화 (또는 학습 모드 비활성화)
```

##### 입력 텍스트에 Byte-Pair Encoding (BPE) 적용하기
```python
tokens = roberta.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
assert roberta.decode(tokens) == 'Hello world!'
```

##### RoBERTa에서 특징(feature) 추출
```python
# 마지막 계층의 특징 추출
last_layer_features = roberta.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# 모든 계층의 특징 추출
all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
```

##### 문장 관계 분류(sentence-pair classification) 태스크에 RoBERTa 사용하기
```python
# MNLI에 대해 미세조정된 RoBERTa 다운로드
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # 평가를 위해 드롭아웃 비활성화

with torch.no_grad():
    # 한 쌍의 문장을 인코딩하고 예측
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 0  # contradiction

    # 다른 문장 쌍을 인코딩하고 예측
    tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
    prediction = roberta.predict('mnli', tokens).argmax().item()
    assert prediction == 2  # entailment
```

##### 새로운 분류층 적용하기
```python
roberta.register_classification_head('new_task', num_classes=3)
logprobs = roberta.predict('new_task', tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
```


### 참고

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding][1]
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach][2]


[1]: https://arxiv.org/abs/1810.04805
[2]: https://arxiv.org/abs/1907.11692
