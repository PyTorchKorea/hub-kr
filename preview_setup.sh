#!/bin/bash
rm -rf pytorch.kr
git clone --recursive https://github.com/PyTorchKorea/pytorch.kr.git -b site --depth 1
cp *.md pytorch.kr/_hub
cp images/* pytorch.kr/assets/images/

