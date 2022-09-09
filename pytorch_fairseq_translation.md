---
layout: hub_detail
background-class: hub-background
body-class: hub
title: Transformer (NMT)
summary: 영어-프랑스어 번역과 영어-독일어 번역을 위한 트랜스포머 모델
category: researchers
image: fairseq_logo.png
author: Facebook AI (fairseq Team)
tags: [nlp]
github-link: https://github.com/pytorch/fairseq/
github-id: pytorch/fairseq
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
order: 2
demo-model-link: https://huggingface.co/spaces/pytorch/Transformer_NMT
---


### 모델 설명

논문 [Attention Is All You Need][1]에 소개되었던 트랜스포머(Transformer)는  
강력한 시퀀스-투-시퀀스 모델링 아키텍처로 최신 기계 신경망 번역 시스템을 가능하게 합니다.

최근, `fairseq`팀은 역번역된 데이터를 활용한 
트랜스포머의 대규모 준지도 학습을 통해 번역 수준을 기존보다 향상시켰습니다.
더 자세한 내용은 [블로그 포스트][2]를 통해 찾으실 수 있습니다.


### 요구사항

전처리 과정을 위해 몇 가지 python 라이브러리가 필요합니다:

```bash
pip install bitarray fastBPE hydra-core omegaconf regex requests sacremoses subword_nmt
```


### 영어 ➡️ 프랑스어 번역

영어를 프랑스어로 번역하기 위해 [Scaling
Neural Machine Translation][3] 논문의 모델을 활용합니다:

```python
import torch

# WMT'14 data에서 학습된 영-불 트랜스포머 모델 불러오기:
en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

# GPU 사용 (선택사항):
en2fr.cuda()

# beam search를 통한 번역:
fr = en2fr.translate('Hello world!', beam=5)
assert fr == 'Bonjour à tous !'

# 토큰화:
en_toks = en2fr.tokenize('Hello world!')
assert en_toks == 'Hello world !'

# BPE 적용:
en_bpe = en2fr.apply_bpe(en_toks)
assert en_bpe == 'H@@ ello world !'

# 이진화:
en_bin = en2fr.binarize(en_bpe)
assert en_bin.tolist() == [329, 14044, 682, 812, 2]

# top-k sampling을 통해 다섯 번역 사례 생성:
fr_bin = en2fr.generate(en_bin, beam=5, sampling=True, sampling_topk=20)
assert len(fr_bin) == 5

# 예시중 하나를 문자열로 변환하고 비토큰화
fr_sample = fr_bin[0]['tokens']
fr_bpe = en2fr.string(fr_sample)
fr_toks = en2fr.remove_bpe(fr_bpe)
fr = en2fr.detokenize(fr_toks)
assert fr == en2fr.decode(fr_sample)
```


### 영어 ➡️ 독일어 번역

역번역에 대한 준지도학습은 번역 시스템을 향상시키는데 효율적인 방법입니다.
논문 [Understanding Back-Translation at Scale][4]에서,
추가적인 학습 데이터로 사용하기 위해 2억개 이상의 독일어 문장을 역번역합니다. 이 다섯 모델들의 앙상블은 [WMT'18 English-German news translation competition][5]의 수상작입니다.

[noisy-channel reranking][6]을 통해 이 접근법을 더 향상시킬 수 있습니다. 
더 자세한 내용은 [블로그 포스트][7]에서 볼 수 있습니다. 
이러한 노하우로 학습된 모델들의 앙상블은 [WMT'19 English-German news
translation competition][8]의 수상작입니다.

앞서 소개된 대회 수상 모델 중 하나를 사용하여 영어를 독일어로 번역해보겠습니다:

```python
import torch

# WMT'19 data에서 학습된 영-독 트랜스포머 모델 불러오기:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')

# 기본 트랜스포머 모델에 접근
assert isinstance(en2de.models[0], torch.nn.Module)

# 영-독 번역
de = en2de.translate('PyTorch Hub is a pre-trained model repository designed to facilitate research reproducibility.')
assert de == 'PyTorch Hub ist ein vorgefertigtes Modell-Repository, das die Reproduzierbarkeit der Forschung erleichtern soll.'
```

교차 번역으로 같은 문장에 대한 의역을 만들 수도 있습니다:
```python
# 영어과 독일어 교차번역:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

paraphrase = de2en.translate(en2de.translate('PyTorch Hub is an awesome interface!'))
assert paraphrase == 'PyTorch Hub is a fantastic interface!'

# 영어-러시아어 교차번역과 비교:
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

paraphrase = ru2en.translate(en2ru.translate('PyTorch Hub is an awesome interface!'))
assert paraphrase == 'PyTorch is a great interface!'
```


### 참고 문헌

- [Attention Is All You Need][1]
- [Scaling Neural Machine Translation][3]
- [Understanding Back-Translation at Scale][4]
- [Facebook FAIR's WMT19 News Translation Task Submission][6]


[1]: https://arxiv.org/abs/1706.03762
[2]: https://code.fb.com/ai-research/scaling-neural-machine-translation-to-bigger-data-sets-with-faster-training-and-inference/
[3]: https://arxiv.org/abs/1806.00187
[4]: https://arxiv.org/abs/1808.09381
[5]: http://www.statmt.org/wmt18/translation-task.html
[6]: https://arxiv.org/abs/1907.06616
[7]: https://ai.facebook.com/blog/facebook-leads-wmt-translation-competition/
[8]: http://www.statmt.org/wmt19/translation-task.html
