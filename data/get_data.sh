#!/usr/bin/env bash

set -e

git clone https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes.git tmp \
    && mv tmp/stage1_train . \
    && rm -rf tmp
git clone https://github.com/yuanqing811/DSB2018_stage1_test.git tmp \
    && mv tmp/stage1_test . \
    && rm -rf tmp
kaggle competitions download \
    -c data-science-bowl-2018 \
    -f stage2_test_final.zip -p . \
    && mkdir stage2_test && cd stage2_test \
    && unzip ../stage2_test_final.zip && cd .. \
    && rm stage2_test_final.zip
