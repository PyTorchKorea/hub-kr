# Model Contributing Guide line
pytorch hub에 모델을 추가하는 Contributing 가이드라인 입니다.

## 기여방법
[PyTorch 한국어 모델 허브 사이트](https://pytorch.kr/hub/)에 새로운 모델을 추가하기 위한 기여 방법입니다

#### 1. 이슈 남기기

(매우 낮은 확률로) 해당 모델이 추가되는 중일 수 있으니, 작업을 진행하기 전에 [본 저장소에 이슈](https://github.com/PyTorchKorea/hub-kr/issues)를 검색하거나 새로 남겨주세요.

해당 문제점에 대한 개선 사항이 이미 논의되었거나 진행 중인 Pull Request를 통해 해결 중일 수 있으니, 새로 이슈를 만드시기 전, 먼저 검색을 해주시기를 부탁드립니다.

이후, 새로 남겨주신 이슈에서 저장소 관리자 및 다른 방문자들이 함께 문제점에 대해 토의하실 수 있습니다. (또는 이미 관련 이슈가 존재하지만 해결 중이지 않은 경우에는 덧글을 통해 기여를 시작함을 알려주세요.)

#### 2. 저장소 복제하기

새로운 모델에 대한 설명을 추가하기 위해 저장소를 복제합니다.
저장소 복제가 처음이시라면 [GitHub의 저장소 복제 관련 도움말](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)을 참조해주세요.


#### 3. ```hubconf.py``` 작성하기

```torch.hub.load```를 통하여 모델을 불러오기 위해서는 해당 모델의 저장소에 ```hubconf.py```가 추가되여야 합니다. 

```python
dependencies = ['torch']


def model_name(*args, **kwargs):
    """
    Docstring : torch.hub.help() 에 나타 날 부분
    """
    # 모델을 정의하고, 사전학습된 가중치를 모델에 불러옵니다
    model = load_model(pretrained = True, **kwargs)
    return model
```

```hubconf.py``` 는 다음의 요소들을 가지고 있고, 조건들을 만족하여야 합니다.

* ```dependencies``` : 모델을 불러오기 위해 필요한 패키지들을 작성해 두는 변수입니다.
* ```docstring``` : 모델을 사용하기 위하여 도움을 주는 문서입니다. ```torch.hub.help``` 를 통하여 출력 됩니다.
* entrypoint 함수들은 모델(nn.Module)이나 원할한 작업을 위한 서포트툴을 반환해야합니다.
* torch.hub.list로 보여지고 싶지 않은 함수들은 접두사에 ```_``` 문자를 붙여 나타나지 않게 할 수 있습니다.
* 사전학습된 가중치들을 불러오기 위해서 ```torch.hub.load_state_dict_from_url()``` 이용하여 url로 부터 불러올 수 있습니다.

더 자세한 내용은 [torch.hub docs](https://pytorch.org/docs/master/hub.html#publishing-models)를 참고하시길 바랍니다.

#### 4. 허브 문서 작성

복사한 hub kr에 ```<repo_owner>_<repo_name>_<title>.md``` 의 형식으로 문서를 생성하고 다음의 [템플릿](https://github.com/PyTorchKorea/hub-kr/blob/master/docs/template.md)에 맞춰 작성합니다.

#### 5. (내 컴퓨터에서) 결과 확인하기

저장소의 최상위 경로에서 `preview_hub.sh` 명령어를 이용하면 코드 실행 없이 `http://127.0.0.1:4000/` 로컬 주소를 활용하여 빌드 결과물을 빠르게 확인하실 수 있습니다.  \
이 과정에서 수정한 문서 상에서 발생하는 오류가 있다면 Markdown 문법을 참고하여 올바르게 고쳐주세요.

#### 6. Pull Request 만들기

번역을 완료한 내용을 복제한 저장소에 Commit 및 Push하고, Pull Request를 남깁니다. \
Pull Request를 만드시기 전에 이 문서에 포함된 [Pull Request 만들기](#Pull-Request-만들기) 부분을 반드시 읽어주세요. \
만약 Pull Request 만들기가 처음이시라면 [GitHub의 Pull Request 소개 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) 및 [복제한 저장소로부터 Pull Request 만들기 도움말](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)을 참조해주세요