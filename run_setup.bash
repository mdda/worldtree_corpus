#!/usr/bin/env bash
set -e

# Get a copy of the desk reference here...
PDF=worldtree_explanation_corpus_v2.1_book_desk_reference_draft.pdf
if [ ! -f "$PDF" ]; then
    #wget http://cognitiveai.org/dist/${PDF}  # Optional, TBH
    echo "Not downloading - too big"
fi

TASK_REPO=tg2021task
if [ ! -d "$TASK_REPO" ]; then
    git clone https://github.com/cognitiveailab/${TASK_REPO}.git
fi

mkdir -p ./predictions

VENV=env38
if [ ! -d "$VENV" ]; then
    virtualenv --system-site-packages ${VENV}
fi
. ./env38/bin/activate

pip3 install -r requirements.txt
pip3 install -r ${TASK_REPO}/requirements.txt   # Already included in this repo
python3 -m spacy download en_core_web_sm   # 12Mb
#python3 -m spacy download en_core_web_lg   # 868Mb For vectorizers.SpacyVectorizer
if [ -d .git ]; then
  nbstripout --install  # Make notebooks source controllable
fi

pushd ${TASK_REPO}
    make dataset   # Downloads ~1Mb, and expands it out into the tables/*.tsv files, etc
popd
