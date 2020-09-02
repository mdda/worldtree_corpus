#!/usr/bin/env bash
set -e

. env3/bin/activate

TASK_REPO=tg2020task

pushd ${TASK_REPO}
    python3 baseline_tfidf.py tables questions.train.tsv > predict.train.txt  # <45secs
    python3 evaluate.py --gold questions.train.tsv predict.train.txt
    # > MAP:  0.24691065411162139

    python3 baseline_tfidf.py tables questions.dev.tsv > predict.dev.txt  # <15secs
    python3 evaluate.py --gold questions.dev.tsv predict.dev.txt
    # > MAP:  0.2550229026109089

    METHOD=baseline_tfidf
    python3 ${METHOD}.py tables questions.test.tsv > predict.test.${METHOD}.txt
    zip predict.test.${METHOD}.zip predict.test.${METHOD}.txt

    echo "Uploadable predictions file : ./${TASK_REPO}/predict.test.${METHOD}.zip"
popd

# Doing this directly from outer repo is simply:
#
# METHOD=rdai_method_1
# python3 ${METHOD}.py ${TASK_REPO}/questions.test.tsv > predict.test.${METHOD}.txt
# zip predict.test.${METHOD}.zip predict.test.${METHOD}.txt
