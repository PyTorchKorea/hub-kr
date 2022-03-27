---
layout: hub_detail
background-class: hub-background
body-class: hub
title: PyTorch-Transformers
summary:  PyTorch implementations of popular NLP Transformers
category: researchers
image: huggingface-logo.png
author: HuggingFace Team
tags: [nlp]
github-link: https://github.com/huggingface/pytorch-transformers.git
github-id: huggingface/transformers
featured_image_1: no-image
featured_image_2: no-image
accelerator: cuda-optional
order: 10
demo-model-link: https://huggingface.co/spaces/pytorch/transformers
---

# 모델 설명


PyTorch-Transformers (이전엔 `pytorch-pretrained-bert`으로 알려짐) 는 자연어 처리(NLP)를 위한 최신식 사전 학습된 모델들을 모아놓은 라이브러리입니다.

라이브러리는 현재 다음 모델들에 대한 파이토치 구현과 사전 학습된 가중치, 사용 스크립트, 변환 유틸리티를 포함하고 있습니다.

1. **[BERT](https://github.com/google-research/bert)** 는 Google에서 발표한 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 논문과 함께 공개되었습니다. (저자: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova)
2. **[GPT](https://github.com/openai/finetune-transformer-lm)** 는 OpenAI에서 발표한 [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) 논문과 함께 공개되었습니다. (저자: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever)
3. **[GPT-2](https://blog.openai.com/better-language-models/)** 는 OpenAI에서 발표한 [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) 논문과 함께 공개되었습니다. (저자: Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei**, Ilya Sutskever**)
4. **[Transformer-XL](https://github.com/kimiyoung/transformer-xl)** 는 Google/CMU에서 발표한 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) 논문과 함께 공개되었습니다. (저자: Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov)
5. **[XLNet](https://github.com/zihangdai/xlnet/)** 는 Google/CMU에서 발표한 [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) 논문과 함께 공개되었습니다. (저자: Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le)
6. **[XLM](https://github.com/facebookresearch/XLM/)** 는 Facebook에서 발표한 [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) 논문과 함께 공개되었습니다. (저자: Guillaume Lample, Alexis Conneau)
7. **[RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)** 는 Facebook에서 발표한 [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) 논문과 함께 공개되었습니다. (저자: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov)
8. **[DistilBERT](https://github.com/huggingface/pytorch-transformers/tree/master/examples/distillation)** 는 HuggingFace에서 게시한 [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b) 블로그 포스팅과 함께 발표되었습니다. (저자: Victor Sanh, Lysandre Debut, Thomas Wolf)

여기에서 사용되는 구성요소들은 `pytorch-transformers` 라이브러리에 있는  `AutoModel` 과 `AutoTokenizer` 클래스를 기반으로 하고 있습니다.

# 요구 사항

파이토치 허브에 있는 대부분의 다른 모델들과 다르게, BERT는 별도의 파이썬 패키지들을 설치해야 합니다.

```bash
pip install tqdm boto3 requests regex sentencepiece sacremoses
```

# 사용 방법

사용 가능한 메소드는 다음과 같습니다:
- `config`: 지정한 모델 또는 경로에 해당하는 설정값(configuration)을 반환합니다.
- `tokenizer`: 지정한 모델 또는 경로에 해당하는 토크나이저(tokenizer)를 반환합니다.
- `model`: 지정한 모델 또는 경로에 해당하는 모델을 반환합니다.
- `modelForCausalLM`: 지정한 모델 또는 경로에 해당하는, 언어 모델링 헤드(language modeling head)가 추가된 모델을 반환합니다.
- `modelForSequenceClassification`: 지정한 모델 또는 경로에 해당하는, 시퀀스 분류기(sequence classifier)가 추가된 모델을 반환합니다.
- `modelForQuestionAnswering`: 지정한 모델 또는 경로에 해당하는, 질의 응답 헤드(question answering head)가 추가된 모델을 반환합니다.

여기의 모든 메소드들은 다음 인자를 공유합니다: `pretrained_model_or_path` 는 반환할 인스턴스에 대한 사전 학습된 모델 또는 경로를 나타내는 문자열입니다. 각 모델에 대해 사용할 수 있는 다양한 체크포인트(checkpoint)가 있고, 자세한 내용은 아래에서 확인하실 수 있습니다:




사용 가능한 모델은 [pytorch-transformers 문서의 pre-trained models 섹션](https://huggingface.co/pytorch-transformers/pretrained_models.html)에 나열되어 있습니다.

# 문서

다음은 각 사용 가능한 메소드들의 사용법을 자세히 설명하는 몇 가지 예시입니다.


## 토크나이저

토크나이저 객체로 문자열을 모델에서 사용할 수 있는 토큰으로 변환할 수 있습니다. 각 모델마다 고유한 토크나이저가 있고, 일부 토큰화 메소드는 토크나이저에 따라 다릅니다. 전체 문서는 [여기](https://huggingface.co/pytorch-transformers/main_classes/tokenizer.html)에서 확인해보실 수 있습니다.

```py
import torch
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # S3 및 캐시에서 어휘(vocabulary)를 다운로드합니다.
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', './test/bert_saved_model/')  # `save_pretrained('./test/saved_model/')`를 통해 토크나이저를 저장한 경우에 로딩하는 예시입니다.
```

## 모델

모델 객체는 `nn.Module` 를 상속하는 모델의 인스턴스입니다. 각 모델은 로컬 파일 혹은 디렉터리나 사전 학습할 때 사용된 설정값(앞서 설명한 `config`)으로부터 저장/로딩하는 방법이 함께 제공됩니다. 각 모델은 다르게 동작하며, 여러 다른 모델들의 전체 개요는 [여기](https://huggingface.co/pytorch-transformers/pretrained_models.html)에서 확인해보실 수 있습니다.

```py
import torch
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')    # S3와 캐시로부터 모델과 설정값을 다운로드합니다.
model = torch.hub.load('huggingface/pytorch-transformers', 'model', './test/bert_model/')  # `save_pretrained('./test/saved_model/')`를 통해 모델을 저장한 경우에 로딩하는 예시입니다.
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', output_attentions=True)  # 설정값을 업데이트하여 로딩합니다.
assert model.config.output_attentions == True
# 파이토치 모델 대신 텐서플로우 체크포인트 파일로부터 로딩합니다. (느림)
config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
model = torch.hub.load('huggingface/pytorch-transformers', 'model', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
```

## 언어 모델링 헤드가 추가된 모델

앞서 언급한, 언어 모델링 헤드가 추가된 `model` 인스턴스입니다.

```py
import torch
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')    # huggingface.co와 캐시로부터 모델과 설정값을 다운로드합니다.
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './test/saved_model/')  # `save_pretrained('./test/saved_model/')`를 통해 모델을 저장한 경우에 로딩하는 예시입니다.
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2', output_attentions=True)  # 설정값을 업데이트하여 로딩합니다.
assert model.config.output_attentions == True
# 파이토치 모델 대신 텐서플로우 체크포인트 파일로부터 로딩합니다. (느림)
config = AutoConfig.from_pretrained('./tf_model/gpt_tf_model_config.json')
model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './tf_model/gpt_tf_checkpoint.ckpt.index', from_tf=True, config=config)
```

## 시퀀스 분류기가 추가된 모델

앞서 언급한, 시퀀스 분류기가 추가된 `model` 인스턴스입니다.

```py
import torch
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased')    # S3와 캐시로부터 모델과 설정값을 다운로드합니다.
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', './test/bert_model/')  # `save_pretrained('./test/saved_model/')`를 통해 모델을 저장한 경우에 로딩하는 예시입니다.
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-uncased', output_attention=True)  # 설정값을 업데이트하여 로딩합니다.
assert model.config.output_attention == True
# 파이토치 모델 대신 텐서플로우 체크포인트 파일로부터 로딩합니다. (느림)
config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
```

## 질의 응답 헤드가 추가된 모델

앞서 언급한, 질의 응답 헤드가 추가된 `model` 인스턴스입니다.

```py
import torch
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-base-uncased')    # S3와 캐시로부터 모델과 설정값을 다운로드합니다.
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', './test/bert_model/')  # `save_pretrained('./test/saved_model/')`를 통해 모델을 저장한 경우에 로딩하는 예시입니다.
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-base-uncased', output_attention=True)  # 설정값을 업데이트하여 로딩합니다.
assert model.config.output_attention == True
# 파이토치 모델 대신 텐서플로우 체크포인트 파일로부터 로딩합니다. (느림)
config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
```

## 설정값

설정값은 선택 사항입니다. 설정값 객체는 모델에 관한 정보, 예를 들어 헤드나 레이어의 개수, 모델이 어텐션(attentions) 또는 은닉 상태(hidden states)를 출력해야 하는지, 또는 모델이 TorchScript에 맞게 조정되어야 하는지 여부에 대한 정보를 가지고 있습니다. 각 모델에 따라 다양한 매개변수를 사용할 수 있습니다. 전체 문서는 [여기](https://huggingface.co/pytorch-transformers/main_classes/configuration.html)에서 확인해보실 수 있습니다.

```py
import torch
config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')  # S3와 캐시로부터 모델과 설정값을 다운로드합니다.
config = torch.hub.load('huggingface/pytorch-transformers', 'config', './test/bert_saved_model/')  # `save_pretrained('./test/saved_model/')`를 통해 모델을 저장한 경우에 로딩하는 예시입니다.
config = torch.hub.load('huggingface/pytorch-transformers', 'config', './test/bert_saved_model/my_configuration.json')
config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased', output_attention=True, foo=False)
assert config.output_attention == True
config, unused_kwargs = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased', output_attention=True, foo=False, return_unused_kwargs=True)
assert config.output_attention == True
assert unused_kwargs == {'foo': False}

# 설정값을 사용하여 모델을 로딩합니다.
config = torch.hub.load('huggingface/pytorch-transformers', 'config', 'bert-base-uncased')
config.output_attentions = True
config.output_hidden_states = True
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', config=config)
# 모델은 이제 어텐션과 은닉 상태도 출력하도록 설정되었습니다.

```

# 사용 예시

다음은 입력 텍스트를 토큰화한 후 BERT 모델에 입력으로 넣어서 계산된 은닉 상태를 가져오거나, 언어 모델링 BERT 모델을 이용하여 마스킹된 토큰들을 예측하는 방법에 대한 예시입니다.

## 먼저, 입력을 토큰화하기

```python
import torch
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"

# 주위에 특수 토큰이 있는 입력을 토큰화합니다. (BERT에서는 처음과 끝에 각각 [CLS]와 [SEP] 토큰이 있습니다.)
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
```

## `BertModel`을 사용하여, 입력 문장을 마지막 레이어 은닉 상태의 시퀀스로 인코딩하기

```python
# 첫번째 문장 A와 두번째 문장 B의 인덱스를 정의합니다. (논문 참조)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# 입력값을 PyTorch tensor로 변환합니다.
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, token_type_ids=segments_tensors)
```

## `modelForMaskedLM`을 사용하여, BERT로 마스킹된 토큰 예측하기

```python
# `BertForMaskedLM`를 통해 예측할 토큰을 마스킹(마스크 토큰으로 변환)합니다.
masked_index = 8
indexed_tokens[masked_index] = tokenizer.mask_token_id
tokens_tensor = torch.tensor([indexed_tokens])

masked_lm_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForMaskedLM', 'bert-base-cased')

with torch.no_grad():
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)

# 예측된 토큰을 가져옵니다.
predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'Jim'
```

## `modelForQuestionAnswering`을 사용하여, BERT로 질의 응답하기

```python
question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

# 형식은 단락이 먼저 주어지고, 그 다음에 질문이 주어지는 형식입니다.
text_1 = "Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ?"
indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

# 시작 및 종료 위치에 대한 로짓(logits)을 예측합니다.
with torch.no_grad():
    out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

# 가장 높은 로짓을 가진 예측을 가져옵니다.
answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1])
assert answer == "puppeteer"

# 또는 시작 및 종료 위치에 대한 교차 엔트로피 손실의 총합을 가져옵니다. (이 코드가 학습 시에 사용되는 경우 미리 모델을 학습 모드로 설정해야 합니다.)
start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
multiple_choice_loss = question_answering_model(tokens_tensor, token_type_ids=segments_tensors, start_positions=start_positions, end_positions=end_positions)
```

## `modelForSequenceClassification`을 사용하여, BERT로 패러프레이즈(paraphrase) 분류하기

```python
sequence_classification_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 'bert-base-cased-finetuned-mrpc')
sequence_classification_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased-finetuned-mrpc')

text_1 = "Jim Henson was a puppeteer"
text_2 = "Who was Jim Henson ?"
indexed_tokens = sequence_classification_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
segments_tensors = torch.tensor([segments_ids])
tokens_tensor = torch.tensor([indexed_tokens])

# 시퀀스 분류를 위한 로짓을 예측합니다.
with torch.no_grad():
    seq_classif_logits = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensors)

predicted_labels = torch.argmax(seq_classif_logits[0]).item()

assert predicted_labels == 0  # MRPC 데이터셋에서, 이는 두 문장이 서로 바꾸어 표현할 수 없다는 것을 뜻합니다.

# 또는 시퀀스 분류에 대한 손실을 가져옵니다. (이 코드가 학습 시에 사용되는 경우 미리 모델을 학습 모드로 설정해야 합니다.)
labels = torch.tensor([1])
seq_classif_loss = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensors, labels=labels)
```
