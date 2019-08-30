import csv

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances
from rank import maybe_concat_texts, BM25Vectorizer, deduplicate_combos, remove_combo_suffix, add_missing_idxs
"""
Dev Results
Experiment parameters                           | spearmanr | MAP
epochs 3, scaling 0                             | 0.61      | 0.486
epochs 3, scaling 1                             | 0.64      | 0.487
epochs 2, scaling 1                             | 0.62      | 0.473
epochs 3, scaling 1, bert_large_masking 1       | 0.09      | 0.009
epochs 3, scaling 1, top_n 128                  | 0.65      | 0.481
epochs 3, scaling 1, top_n 128, bert_large 1    | 0.63      | 0.478
epochs 3, scaling 0, top_n 64, shuffle          | 0.62      | 0.501
epochs 3, scaling 0, top_n 64(dev1024), shuffle | 0.34      | 0.507
epochs 3, scaling 0, top_n 1024(dev1024), shuffle 0.06      | 0.490
epochs 3, scaling 0, top_n 64, shuffle, scores**4 0.63      | 0.487
"""


def make_score_data(
    df: pd.DataFrame,
    df_exp: pd.DataFrame,
    rankings: list,
    top_n: int = 64,
) -> pd.DataFrame:
  q = maybe_concat_texts(df["lemmas"].tolist())
  e = maybe_concat_texts(df_exp["lemmas"].tolist())
  vec = BM25Vectorizer()
  vec.fit(q + e)
  vectors_e = vec.transform(e)

  # Gold explanations
  def concat_exp_text(exp_idxs):
    def concat_lst(lst):
      return " ".join(lst)

    return " ".join(df_exp.lemmas.iloc[exp_idxs].apply(concat_lst).tolist())

  e_gold = df.exp_idxs.apply(concat_exp_text)
  vectors_e_gold = vec.transform(e_gold)
  matrix_dist_gold = cosine_distances(vectors_e_gold, vectors_e)
  top_ranks = [ranks[:top_n] for ranks in rankings]
  top_dists = [
      matrix_dist_gold[i][top_ranks[i]] for i in range(len(top_ranks))
  ]

  data = []
  for i in range(len(top_ranks)):
    text_q = df.q_reformat.iloc[i]
    texts_e = df_exp.text.iloc[top_ranks[i]].tolist()
    for j in range(top_n):
      data.append([text_q, texts_e[j], top_dists[i][j]])

  df_scores = pd.DataFrame(data, columns=["text_q", "text_e", "score"])
  print(df_scores.shape)
  return df_scores


def preproc_trn_data(df: pd.DataFrame) -> pd.DataFrame:
  """
  Three reasons to remove qe pairs with score == 1.0:
  1. Questions without explanations will always result in 1.0
  2. Valid qe pairs with 1.0 means the explanation is completely unrelated
      which is too easy for the model
  3. They skew the label/label distribution
  """
  print("Preprocessing train bert data (df_scores)")
  old_length = len(df)
  df = df[~(df.score == 1.0)]
  print(f"Dropping irrelevant explanations ({old_length} -> {len(df)})")
  df = df.sample(frac=1).reset_index(drop=True)  # shuffle
  print("Plotting histrogram distribution of scores")
  df.score.hist(bins=50)
  return df


def read_predict_txt(
    df: pd.DataFrame,
    df_exp: pd.DataFrame,
    in_file: str,
) -> list:
  qid2idx = {qid: idx for idx, qid in enumerate(df.questionID.tolist())}
  uid2idx = {uid: idx for idx, uid in enumerate(df_exp.uid.tolist())}
  ranks = [[] for _ in range(len(df))]
  orig_exp_set = set(df_exp.uid.apply(remove_combo_suffix).tolist())
  orig_exp_idxs = df_exp[df_exp.uid.isin(orig_exp_set)].index.tolist()

  with open(in_file) as file:
    for line in file:
      qid, uid = line.strip().split()
      ranks[qid2idx[qid]].append(uid2idx[uid])
  assert all([len(lst) > 0 for lst in ranks])
  ranks = [add_missing_idxs(r, idxs_sample=orig_exp_idxs) for r in ranks]
  ranks = deduplicate_combos(ranks, df_exp)
  ranks = [np.asarray(r) for r in ranks]
  return ranks


def rerank_ranks(
    df: pd.DataFrame,
    ranks: list,
    scores: np.ndarray,
    do_average_ranks: bool = True,
) -> list:
  def copy(nested_list):
    return [list(lst) for lst in nested_list]

  ranks_old = copy(ranks)
  ranks = copy(ranks)
  assert len(df) == len(ranks)
  scores = np.split(scores, len(df))
  top_n = len(scores[0])

  for i in range(len(df)):
    idxs_sort_front = np.argsort(scores[i])
    old_length = len(ranks[i])
    front = np.asarray(ranks[i][:top_n])
    front_rerank = list(front[idxs_sort_front])
    back = list(ranks[i][top_n:])
    assert len(front_rerank) + len(back) == old_length
    ranks[i] = front_rerank + back

  ranks = [np.asarray(lst) for lst in ranks]  # back to orig dtypes

  if do_average_ranks:
    ranks = average_ranks(ranks_old, ranks)  # Extra 1% MAP by ensembling
  return ranks


def average_ranks(ranks1: list, ranks2: list) -> list:
  assert len(ranks1) == len(ranks2)

  def process(r1: list, r2: list) -> np.ndarray:
    assert len(r1) == len(r2)
    assert set(r1) == set(r2)
    d = {item: [] for item in r1}
    for r in [r1, r2]:
      for rank, item in enumerate(r):
        d[item].append(rank)

    d = {item: np.mean(r) for item, r in d.items()}
    return np.asarray(sorted(d.keys(), key=lambda item: d[item]))

  return [process(r1, r2) for r1, r2 in zip(ranks1, ranks2)]


def _read_tsv(input_file: str, quotechar: str = None) -> list:
  """Reads a tab separated value file."""
  with tf.gfile.Open(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    lines = []
    for line in reader:
      lines.append(line)
    return lines


def read_tsv(file: str, first_line_header: bool = True) -> pd.DataFrame:
  lines = _read_tsv(file)
  if first_line_header:
    df = pd.DataFrame(lines[1:], columns=lines[0])
  else:
    df = pd.DataFrame(lines)
  return df


def read_preds(in_file: str) -> pd.DataFrame:
  df_pred = read_tsv(in_file, first_line_header=False)
  df_pred[0] = df_pred[0].apply(lambda x: float(x))
  print(df_pred.shape)
  return df_pred
