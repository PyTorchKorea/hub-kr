# PyTorch 한국어 모델 허브 번역 기여하기

PyTorch 한국어 모델 허브 저장소에 방문해주셔서 감사합니다. 이 문서는 PyTorch 한국어 모델 허브에 기여하는 방법을 안내합니다.


## 기여하기 개요

[본 저장소](https://github.com/PyTorchKorea/hub-kr)는 [PyTorch 공식 허브](https://pytorch.org/hub/)를 번역하는 프로젝트를 위한 곳으로,
[Pytorch 공식 허브 저장소](https://github.com/pytorch/hub)의 내용을 비정기적으로 반영하고, 번역 및 개선합니다.

크게 다음과 같은 기여 방법이 있습니다.

* [1. 오탈자를 수정하거나 번역을 개선하는 기여](#1-오탈자를-수정하거나-번역을-개선하는-기여)
  * [PyTorch 한국어 모델 허브 사이트](https://pytorch.kr/hub/)에서 발견한 오탈자를 [본 저장소](https://github.com/PyTorchKorea/hub-kr)에서 고치는 기여입니다.
* [2. 번역되지 않은 허브 모델을 번역하는 기여](#2-번역되지-않은-허브-모델을-번역하는-기여)
  * [PyTorch 한국어 모델 허브 사이트](https://pytorch.kr/hub/)에 아직 번역되지 않은 모델 허브를 번역하는 기여입니다.
* [3. 2로 번역된 문서를 리뷰하는 기여](#3-2로-번역된-문서를-리뷰하는-기여) :star:
  * [본 저장소에 Pull Request된 허브 문서](https://github.com/PyTorchKorea/hub-kr/pulls)를 리뷰하는 기여입니다.

기여 및 리뷰 시 [행동 강령](https://github.com/PyTorchKorea/.github/blob/master/CODE_OF_CONDUCT.md)을 지켜주시면 감사하겠습니다.

## 기여 결과물의 라이선스 동의

PyTorch 한국어 모델 허브는 [Pytorch 공식 허브 저장소](https://github.com/pytorch/hub)와 동일한 [BSD 3항 라이선스](https://github.com/PyTorchKorea/pytorch.kr/blob/master/LICENSE)를 따릅니다. \
따라서 기여하신 모든 내용에 [BSD 3항 라이선스](https://github.com/PyTorchKorea/pytorch.kr/blob/master/LICENSE)가 적용됨을 인지하시고 동의하시는 경우에만 아래 문서 내용과 같이 기여해주세요.


## 기여하기 절차

모든 기여는 [본 저장소에 이슈](https://github.com/PyTorchKorea/hub-kr/issues)를 남긴 후 [Pull Request를 보내는 것](https://github.com/PyTorchKorea/hub-kr/pulls)으로 합니다. \
이 과정을 통해 Pull Request를 위한 Commit을 만들기 전에 이슈를 통해 해당 내용에 기여가 필요한지 여부를 확인하고 협의하셔야 합니다. \
(물론 이슈를 남기셨다고 해서 반드시 해당 문제를 개선하셔야 하는 것은 아니니, 마음 편히 이슈를 남겨주세요. :))

### Pull Request 만들기

#### Pull Request 만들기 전 : 주의사항

* 하나의 commit, branch, Pull Request(PR)에는 하나의 변경 사항만 담아주세요.
  * 여러 수정사항에 대해서는 각각 다른 branch에서 작업하신 뒤, 새로운 PR을 만들어주세요.
  * 새로운 branch가 아닌, 이미 PR를 만드셨던 branch에 추가 commit 시에는 이전 commit들과 함께 Pull Request가 생성됩니다.
* Pull Request를 만들기 전 문법 오류나 깨진 글자는 없는지 확인해주세요.
  * 기본적인 문법은 Markdown 문법을 지키면서 작성해주세요.  
  * 이미 번역된 문서들을 참고하셔도 좋습니다.  
  * 번역 후에는 (내 컴퓨터에서) 빌드를 한 후, 문법 오류를 확인해주세요.  
* 오류가 많거나 다른 PR의 commit이 섞여 있는 경우 해당 PR은 관리자가 닫을 수 있으니 주의해주세요.
* Commit 메시지 작성 규칙을 지켜주세요.
	* 새로운 번역 작성 시는 "[번역]:ResNet 모델"
	* 번역에 대한 수정을 반영할 때는 "[Fix]:오타 수정"

#### Pull Request 만들기 : 생성하기

* `라이선스 동의` 체크하기 ✅
  * 기여해주신 내용을 더 많은 분이 참고 / 개선 / 변경할 수 있게 라이선스 적용에 동의해주세요.
  * 동의를 거부하실 수 있으나, 이 경우 해당 PR의 내용의 자유로운 사용이 어렵기 때문에 리뷰 및 반영은 진행하지 않습니다.
* PR 내용에 관련 이슈 번호 적어주기 🔢
  * 논의된 내용이 있다면 참고할 수 있도록 어떠한 이슈로부터 생성한 PR인지 알려주세요.
* PR 종류 선택하기
  * 리뷰어에게 어떤 종류의 PR인지 알려주세요.
* PR 설명하기
  * 이 PR을 통해 어떠한 것들이 변경되는지 알려주세요.
* **Tip**: 만약 문서가 방대해서 중간 피드백이 필요하다면 Draft PR 기능을 사용할 수 있습니다.
  * 자세한 내용은 [GitHub Blog](https://github.blog/2019-02-14-introducing-draft-pull-requests/)의 글을 참고해주세요.

#### Pull Request 만든 후 : 리뷰를 받았을 때

* 리뷰 내용에 대한 추가 의견이 있을 경우 해당 리뷰에 댓글로 의견을 주고받습니다.
  * 번역한 문서의 내용은 번역자가 가장 잘 알고 있으므로 리뷰어의 의견에 반드시 따라야 하는 것은 아닙니다.
  * 하지만 번역 실수나 오류, 잘못된 Markdown 문법에 대한 내용은 가급적 반영해주시기를 부탁드립니다.
  * 다른 문서들과의 일관성, 이해를 위해 추가로 요청드리는 내용들도 있을 수 있으니 감안해주세요.
* 변경 사항을 고치기로 하였다면, Pull Request를 만든 원본 저장소 / branch에 추가 commit을 합니다.
  * 리뷰 결과를 반영한 경우 `Resolve Conversation` 버튼을 눌러 리뷰어에게 알립니다.

### Pull Request 리뷰하기

* 리뷰 전 (TRANSLATION_GUIDE.md - TBD) 문서를 읽고 리뷰해주세요.
* 특히 다음의 내용들을 유의해주세요.
  * 번역된 용어들이 용어집에 맞게 사용되었는지 확인합니다.
  * 번역된 내용에 오탈자가 있는지 확인해 봅니다.
  * 부자연스러운 내용이 있다면 좀 더 나은 번역으로 제안하여 봅니다.
  * Markdown 문법에 맞게 잘 작성되어있는지 확인해 봅니다.
* 말하려는 내용이 이미 다른 댓글에 있다면 공감 이모지 눌러주세요.


## (기여 종류에 따른) 기여 방법

### 1. 오탈자를 수정하거나 번역을 개선하는 기여
<details>
  <summary>펼치기</summary>

[PyTorch 한국어 모델 허브 사이트](https://pytorch.kr/hub/)에서 발견한 오탈자를 고치는 기여 방법입니다.

#### 1-1. 이슈 남기기

(매우 낮은 확률로) 해당 오탈자가 의도한 것일 수 있으니, 해당 문제점을 고친 Pull Request를 생성하기 전에 [본 저장소에 이슈](https://github.com/PyTorchKorea/hub-kr/issues)를 검색하거나 새로 남겨주세요.

해당 문제점에 대한 개선 사항이 **이미 논의되었거나 진행 중인 Pull Request를 통해 해결 중일 수 있으니, 새로 이슈를 만드시기 전, 먼저 검색**을 해주시기를 부탁드립니다. 

이후, 새로 남겨주신 이슈에서 저장소 관리자 및 다른 방문자들이 함께 문제점에 대해 토의하실 수 있습니다. (또는 이미 관련 이슈가 존재하지만 해결 중이지 않은 경우에는 댓글을 통해 기여를 시작함을 알려주세요.)

#### 1-2. 저장소 복제하기

오탈자를 수정하기 위해 저장소를 복제합니다. \
저장소 복제가 처음이시라면 [GitHub의 저장소 복제 관련 도움말](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)을 참조해주세요.


#### 1-3. 오탈자 수정하기

위에서 찾은 원본 허브 문서를 Markdown 문법에 맞춰 수정합니다. \
Markdown 문법에 익숙하지 않은 경우, 다른 허브 문서의 원본 문서와 빌드 결과물을 비교해보면서 빌드 결과물을 예상할 수 있습니다.

#### 1-4. (내 컴퓨터에서) 결과 확인하기

저장소의 최상위 경로에서 `preview_hub.sh` 명령어를 이용하면 코드 실행 없이 `http://127.0.0.1:4000/` 로컬 주소를 활용하여 빌드 결과물을 빠르게 확인하실 수 있습니다.  

빌드를 위한 자세한 과정은 [Window_build.md](https://github.com/PyTorchKorea/hub-kr/blob/master/Window_build.md)와 [파이토치 허브 README.md](https://github.com/PyTorchKorea/hub-kr)를 참고해주시길 바랍니다.

#### 1-5. Pull Request 만들기

수정을 완료한 내용을 복제한 저장소에 Commit 및 Push하고, Pull Request를 남깁니다. \
Pull Request를 만드시기 전에 이 문서에 포함된 [Pull Request 만들기](#Pull-Request-만들기) 부분을 반드시 읽어주세요. \
만약 Pull Request 만들기가 처음이시라면 [GitHub의 Pull Request 소개 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) 및 [복제한 저장소로부터 Pull Request 만들기 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)을 참조해주세요.

</details>

### 2. 번역되지 않은 허브 모델을 번역하는 기여
<details>
  <summary>펼치기</summary>

[PyTorch 한국어 모델 허브 사이트](https://pytorch.kr/hub/)에 아직 번역되지 않은 모델 허브을 번역하는 기여 방법입니다.

#### 2-1. 이슈 남기기

(매우 낮은 확률로) 해당 허브가 번역 중일 수 있으니, 번역 전에 Pull Request를 생성하기 전에 [본 저장소에 이슈](https://github.com/PyTorchKorea/hub-kr/issues)를 검색하거나 새로 남겨주세요.

해당 허브에 대한 **번역이 이미 논의되었거나 Pull Request를 통해 진행 중일 수 있으니, 새로 이슈를 만드시기 전, 먼저 검색**을 해주시기를 부탁드립니다. \
이후, 새로 남겨주신 이슈에서 저장소 관리자 및 다른 방문자들이 함께 번역 진행에 대해 토의하실 수 있습니다. \
(또는 이미 관련 이슈가 존재하지만 번역 중이지 않은 것처럼 보이는 경우에는 댓글을 통해 기여를 시작함을 알려주세요.)

#### 2-2. 저장소 복제하기

신규 모델 허브을 번역하기 위해 저장소를 복제합니다. \
저장소 복제가 처음이시라면 [GitHub의 저장소 복제 관련 도움말](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)을 참조해주세요.

#### 2-3. 원본 경로 / 문서 찾기

허브 모델 번역을 위해서는 [PyTorch 한국어 모델 허브 사이트](https://pytorch.kr/hub/)의 모델 주소로부터 원본 문서를 찾아야합니다. \
모델 주소에서 `https://pytorch.kr/hub/` 뒷 부분이 문서 이름 입니다. 이 문서는 `https://github.com/PyTorchKorea/hub-kr`에 `.md` 확장자로 존재합니다. \
예를 들어, 파이토치 허브의 YOLOv5 모델 경로가 'https://pytorch.kr/hub/ultralytics_yolov5/' 일 때, 'https://github.com/PyTorchKorea/hub-kr'에 있는 'ultralytics_yolov5.md' 파일이 원본 문서입니다.

#### 2-4. 허브 번역하기

위에서 찾은 원본 허브 문서를 Markdown 문법에 맞춰 번역합니다. \
번역 중 번역 용어에 대해서는 다른 모델 허브 문서를 참조하시거나, `2-1`에서 남긴 이슈의 댓글을 통해 토의하실 수 있습니다. \
Markdown 문법에 익숙하지 않은 경우, 다른 허브 원본 문서와 빌드 결과물을 비교해보면서 빌드 결과물을 예상할 수 있습니다.

#### 2-5. (내 컴퓨터에서) 결과 확인하기

저장소의 최상위 경로에서 `preview_hub.sh` 명령어를 이용하면 코드 실행 없이 `http://127.0.0.1:4000/` 로컬 주소를 활용하여 빌드 결과물을 빠르게 확인하실 수 있습니다.  \
이 과정에서 수정한 문서 상에서 발생하는 오류가 있다면 Markdown 문법을 참고하여 올바르게 고쳐주세요. \

#### 2-6. Pull Request 만들기

번역을 완료한 내용을 복제한 저장소에 Commit 및 Push하고, Pull Request를 남깁니다. \
Pull Request를 만드시기 전에 이 문서에 포함된 [Pull Request 만들기](#Pull-Request-만들기) 부분을 반드시 읽어주세요. \
만약 Pull Request 만들기가 처음이시라면 [GitHub의 Pull Request 소개 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) 및 [복제한 저장소로부터 Pull Request 만들기 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)을 참조해주세요

</details>

### 3. Pull Request에 대해 리뷰하는 기여
<details>
  <summary>펼치기</summary>

[본 저장소에 Pull Request된 허브 문서](https://github.com/PyTorchKorea/hub-kr/pulls)를 리뷰하는 기여입니다.

Pull Request된 문서의 오탈자 수정, Markdown 문법 오류 또는 잘못 번역된 내용을 개선하는 기여로, 가장 기다리고 있는 기여 방식입니다. :pray: \
Pull Request를 리뷰하시기 전에 이 문서에 포함된 [Pull Request 리뷰하기](#Pull-Request-리뷰하기) 부분을 반드시 읽어주세요. \
만약 PR 리뷰가 익숙하지 않으시다면 [GitHub의 Pull Request 리뷰 관련 도움말](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-request-reviews)을 참조해주세요.

</details>