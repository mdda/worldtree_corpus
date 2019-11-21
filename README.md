# Language Model Assisted Explanation Generation

### Multi-Hop Inference Explanation Regeneration 

**\*\*\*\*\* New Oct 11th, 2019: Paper & Slides Release \*\*\*\*\***

Attached [Google Slides presentation](https://docs.google.com/presentation/d/1_YB4M3PRQjGzL9ifAcOzDIEWOlsDgALNmS3aP1hEzE0/edit?usp=sharing) for [Tensorflow & Deep Learning SG Meetup](https://www.meetup.com/TensorFlow-and-Deep-Learning-Singapore/events/265374455/?_xtd=gqFyqTE5NTk2MTk3NqFwpmlwaG9uZQ&from=ref)

Released [conference paper submission](https://arxiv.org/abs/1911.08976) for EMNLP.

**\*\*\*\*\* New August 31st, 2019: Code Release \*\*\*\*\***

Released code implementation of the submission for the shared task of the EMNLP [TextGraphs 2019 Workshop](https://sites.google.com/view/textgraphs2019/home).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mdda/worldtree_corpus/blob/textgraphs/TextGraphs_Workshop_Code.ipynb)

## Introduction

The Explanation Regeneration shared task asked participants to develop methods to reconstruct gold explanations for elementary science questions (Clark et al., 2018), using a new corpus of gold explanations (Jansen et al., 2018) that provides supervision and instrumentation for this multi-hop inference task. Each explanation is represented as an “explanation graph”, a set of atomic facts (between 1 and 16 per explanation, drawn from a knowledge base of 5,000 facts) that, together, form a detailed explanation for the reasoning required to answer and explain the resoning behind a question. Linking these facts to achieve strong performance at rebuilding the gold explanation graphs requires methods to perform multi-hop inference - which has been shown to be far harder than inference of smaller numbers of hops (Jansen, 2018), particularly for the case here, where there is considerable uncertainty (at a lexical level) of how individual explanations logically link somewhat ‘fuzzy’ graph nodes.

To give a few numbers, here are the results on the
[TextGraphs WorkShop Shared Task Competition](https://competitions.codalab.org/competitions/20150):

Method | Test MAP
------------------------------------- | :------:
Original leaderboard submission (August 9th 2019) | 0.4017
OptimizedTFIDF           | 0.4274
IterativeTFIDF       | 0.4576
IterativeTFIDF + BERT re-ranking    | **0.4771**

## Ranking
This example code computes a ranking over all facts in the database and outputs a breakdown of MAP scores over the respective sentence roles.

```shell
python textgraphs/run_ranking.py \
--path_data={path_data} \
```

## Notebooks
For more details on the methods described in the paper, please refer to TextGraphs_Workshop_Code.ipynb

To view the code for our original leaderboard submission, please refer to TextGraphs_SharedTask_Submission.ipynb
