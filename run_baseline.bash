#!/usr/bin/env bash
set -e

. env3/bin/activate

pushd tg2020task
    python3 baseline_tfidf.py tables questions.train.tsv > predict.train.txt  # <45secs
    python3 evaluate.py --gold questions.train.tsv predict.train.txt
    # > MAP:  0.24691065411162139

    python3 baseline_tfidf.py tables questions.dev.tsv > predict.dev.txt  # <15secs
    python3 evaluate.py --gold questions.dev.tsv predict.dev.txt
    # > MAP:  0.2550229026109089

    python3 baseline_tfidf.py tables questions.test.tsv > predict.txt
    zip predict-tfidf-test.zip predict.txt
popd