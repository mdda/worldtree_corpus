#!/usr/bin/env bash
set -e
python3 baseline_tfidf.py tables questions.dev.tsv > predict.txt
python3 evaluate.py --gold questions.dev.tsv predict.txt
python3 baseline_tfidf.py tables questions.test.tsv > predict.txt
zip predict-tfidf-test.zip predict.txt