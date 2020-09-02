#!/usr/bin/env bash
set -e

# Get a copy of the desk reference here...
PDF=worldtree_explanation_corpus_v2.1_book_desk_reference_draft.pdf
if [ ! -f "$PDF" ]; then
    wget http://cognitiveai.org/dist/${PDF}
fi

TASK_REPO=tg2020task
if [ ! -f "$PDF" ]; then
    git clone https://github.com/cognitiveailab/${TASK_REPO}.git
fi

VENV=env3
if [ ! -f "$PDF" ]; then
    virtualenv --system-site-packages ${VENV}
fi
. env3/bin/activate

pip3 install -r requirements.txt
pip3 install -r ${TASK_REPO}/requirements.txt

pushd ${TASK_REPO}
    make dataset
popd
