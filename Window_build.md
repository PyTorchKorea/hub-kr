## Pytorch-hub-kr build in Window 10  

- Author : [Taeyoung96](https://github.com/Taeyoung96)  

Window 10 환경에서 `Pytorch-hub-kr` 빌드를 위한 환경 구축을 하는 방법에 대해 소개해 드리겠습니다.  

환경 설정을 진행하면서 주관적으로 도움을 받았던 링크들을 공유하면서 글을 작성하도록 하겠습니다.  

[Git bash](https://git-scm.com/downloads)에서 Test를 진행했습니다.  

환경 설정 완료 후, `./preview_hub_window.sh`를 command 창에 입력해주시면 빌드 및 미리보기를 할 수 있습니다. 

### 환경 설정  
- Ruby  
- node.js  
- yarn  
- Make command in Window 10  

❗️ 환경 설정 시 각각의 버전을 아래 명시된 버전과 동일하게 맞추면 수월하게 빌드를 진행할 수 있습니다!   

1. Ruby 설치  

[RubyInstaller](https://rubyinstaller.org/downloads/archives/)에서 `Ruby+Devkit 2.7.4-1 (x64)`를 다운 받아 설치합니다.  
(`Ruby+Devkit 2.7.4-1 (x86)`의 경우 Test를 해보지는 않은 상태입니다. 다만 자신의 윈도우 운영체제 비트수에 맞추어 설치를 진행해야 합니다.)  
`ruby -v`로 버전 확인이 가능합니다. `2.7.4` 버전을 설치해야 합니다.  

- ruby 설치 시 참고한 링크 : [[Ruby] 루비 설치하기(Windows 10/윈도우 10) / 예제 맛보기](https://junstar92.tistory.com/5)  
    [How to install RubyGems in Windows?](https://www.geeksforgeeks.org/how-to-install-rubygems-in-windows/)  

2. bundler 설치  

git bash command 창에 `gem install bundler -v 2.3.13`로 명령어를 실행하여 bundler를 설치해주세요.   
버전은 `2.3.13`으로 설치했습니다.  

3. node.js 설치  

node.js 설치는 아래 링크를 참고하여 설치했습니다. 버전은 `16.13.2`입니다.  
nvm 설치시 관리자의 권한으로 `git bash`를 실행해야 합니다.  

- node.js 설치 시 참고한 링크 : [윈도우 node.js 설치하기](https://kitty-geno.tistory.com/61)  
    [node.js와 npm 최신 버전으로 업데이트하기 (window 윈도우)](https://cheoltecho.tistory.com/15)  
    [Access Denied issue with NVM in Windows 10](https://stackoverflow.com/questions/50563188/access-denied-issue-with-nvm-in-windows-10)  

4. yarn 설치  

git bash command 창에 `npm install --global yarn`로 yarn를 설치해주세요.  버전은 `1.22.19`입니다.  

5. Window 10에서 make 명령어 사용  

[ezwinports](https://sourceforge.net/projects/ezwinports/)를 설치해야 합니다. 아래의 참고자료를 참고하여 설치를 진행해주세요.  

- 참고자료 : [MINGW64 "make build" error: "bash: make: command not found"](https://stackoverflow.com/questions/36770716/mingw64-make-build-error-bash-make-command-not-found)  

만약 ezwinports를 통해 make를 설치했음에도 불구하고 make 명령어를 사용할 수 없다면,
chocolatey 를 설치하여 make 명령어를 사용할 수 있습니다.    

- 참고자료 : [chocolatey 설치, 윈도우10에서 sudo,make 명령어 사용하기](https://jie0025.tistory.com/72)  