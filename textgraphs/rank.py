import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import pipeline, feature_extraction, metrics
from tqdm import tqdm

import evaluate_with_role_breakdown as evaluate
from bm25 import BM25Transformer


class MyBM25Transformer(BM25Transformer):
  """
  To be used in sklearn pipeline, transformer.fit()
  needs to be able to accept a "y" argument
  """
  def fit(self, x, y=None):
    super().fit(x)


class BM25Vectorizer(feature_extraction.text.TfidfVectorizer):
  """
  Drop-in, slightly better replacement for TfidfVectorizer
  Best results if text has already gone through stopword removal and lemmatization
  """
  def __init__(self):
    self.vec = pipeline.make_pipeline(
        feature_extraction.text.CountVectorizer(binary=True),
        MyBM25Transformer(),
    )
    super().__init__()

  def fit(self, raw_documents, y=None):
    return self.vec.fit(raw_documents)

  def transform(self, raw_documents, copy=True):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=FutureWarning)
      return self.vec.transform(raw_documents)


def maybe_concat_texts(texts: list) -> list:
  if type(texts[0]) == str:
    return texts
  elif type(texts[0]) == list:
    return [" ".join(sublst) for sublst in texts]
  else:
    raise TypeError(f"Unknown data type: {type(texts[0])}")


def repeat(seed: list, idx2nns: dict) -> dict:
  """
  Start with question
  Get seed, n most relevant
  Add to ranking dict with idx
  For each item in ranking dict, add the nearest neighbour to ranking dict that is not already inside
  Repeat until no more change
  """
  ranking = {item: idx for idx, item in enumerate(seed)}
  idx2nns = dict(idx2nns)

  while True:
    old_ranking = dict(ranking)
    #         print(len(old_ranking), old_ranking)

    for idx in old_ranking.keys():
      while True:
        if len(idx2nns[idx]) == 0:
          break
        n = idx2nns[idx].pop(0)
        if n not in ranking:
          ranking[n] = len(ranking)
          break
    if len(ranking) == len(old_ranking):
      break

  # low rank -> high importance -> high score
  scores = {k: -v for k, v in ranking.items()}
  return scores


def recurse(
    seed: list,
    scores: dict,
    idx2nns: dict,
    iteration: int,
    n: int = 100,
    max_iter: int = 2,
) -> None:
  """
  Start with question
  Get n most relevant/similar
  +1 point each
  Recurse up to iter times 
  Final rank on points
  (MAP=0.04)
  """
  if iteration < max_iter:
    for idx in seed[:n]:
      if idx in scores:
        scores[idx] += 1
      else:
        scores[idx] = 1

    for idx in seed[:n]:
      if scores[idx] > 1:
        continue
      new_seed = idx2nns[idx][1:]  # skip nearest (itself)
      recurse(new_seed, scores, idx2nns, iteration + 1, n, max_iter)


def simple_nn_ranking(
    df: pd.DataFrame,
    df_exp: pd.DataFrame,
    idx: int,
    annoy_index=None,
) -> list:
  # 1000x takes 3.4s
  return annoy_index.get_nns_by_vector(df.embedding.iloc[idx], n=len(df_exp))


def recurse_concat(
    x: sparse.csr_matrix,
    vectors_e: sparse.csr_matrix,
    seq: list,
    maxlen: int = 128,
    top_n: int = 1,
    scale: float = 1.25,
    idx2idx_canon: dict = None,
) -> None:
  """
  Given a text, find the nearest n explanation vectors
  For each, exponentially de-scale and then perform maximum
      Recurse

  default (maxlen=32, top_n=1, scale=2)  -> 0.457
  (scale ** (len(seq) + 1)               -> -0.08
  scale = 1.5 (default=2)                -> +0.05
  scale = 1.25 (default=2)               -> +0.09
  scale = 1.125 (default=2)              -> +0.07
  e = X_e[idx] / (len(seq) + 1)          -> +0.04
  maxlen=64, top_n=2                     -> +0.00
  """
  if len(seq) < maxlen:
    matrix_dist = metrics.pairwise.cosine_distances(x, vectors_e)
    ranks = [np.argsort(distances) for distances in matrix_dist]
    assert len(ranks) == 1
    rank = ranks[0]

    seen = set(seq)
    count = 0
    for idx in rank:
      if count == top_n:
        break

      idx_canon = idx if idx2idx_canon is None else idx2idx_canon[idx]
      if idx_canon not in seen:
        e = vectors_e[idx] / (scale**len(seq))
        new = x.maximum(e)
        seq.append(idx_canon)
        count += 1
        recurse_concat(new, vectors_e, seq)


def get_idx2idx_canon(df_exp: pd.DataFrame) -> dict:
  """
  Slightly improves MAP by 0.0006 when using recursive tfidf and explanation_combos
  """
  uids_canon = df_exp.uid.apply(remove_combo_suffix).tolist()
  idx2uid_canon = {idx: uid for idx, uid in enumerate(uids_canon)}
  e_canon = df_exp[df_exp.uid.isin(set(uids_canon))]
  print("Num canonical explanations:", len(e_canon))
  uid2idx_canon = {uid: idx for idx, uid in zip(e_canon.index, e_canon.uid)}
  return {
      idx: uid2idx_canon[idx2uid_canon[idx]]
      for idx in idx2uid_canon.keys()
  }


def recurse_concat_helper(
    x: sparse.csr_matrix,
    vectors_e: sparse.csr_matrix,
    idx2idx_canon: dict,
) -> list:
  seq = []
  recurse_concat(x, vectors_e, seq, idx2idx_canon=idx2idx_canon)
  return seq


def get_recurse_concat_ranks(df: pd.DataFrame, df_exp: pd.DataFrame) -> list:
  field_q = field_e = "lemmas"
  q = maybe_concat_texts(df[field_q].tolist())
  e = maybe_concat_texts(df_exp[field_e].tolist())
  vec = BM25Vectorizer()
  vec.fit(q + e)
  vectors_q = vec.transform(q)
  vectors_e = vec.transform(e)
  idx2idx_canon = get_idx2idx_canon(df_exp)

  assert len(df) == len(q) == vectors_q.shape[0]

  _recurse_concat_helper = partial(
      recurse_concat_helper,
      vectors_e=vectors_e,
      idx2idx_canon=idx2idx_canon,
  )

  with Pool() as pool:
    # About 50% faster than without multiprocessing
    xs = [vectors_q[i] for i in range(len(df))]
    results = pool.imap(_recurse_concat_helper, xs, chunksize=10)
    ranks = list(tqdm(results, total=len(xs)))
  return ranks


def get_tfidf_ranking(
    df: pd.DataFrame,
    df_exp: pd.DataFrame,
    field_q: str = "q_reformat",
    field_e: str = "text",
    mode: str = "bm25",
) -> list:
  """
  Rank explanation sentences by cosine distance of tfidf vector from question text

  "q_reformat" instead of "Question"      MAP +0.07 (0.24 -> 0.31)
  tfidf stop_words="english"              MAP +0.01 (0.31 -> 0.32)
  field_q and field_e = "lemmas"          MAP +0.08 (0.32 -> 0.40)
  tfidf binary=True                       MAP +0.03 (0.40 -> 0.43)
  tfidf ngram_range=(1,2)                 MAP -0.05 (0.43 -> 0.38)
  tfidf ngram_range=(1,3)                 MAP -0.07 (0.43 -> 0.36)
  tfidf lowercase=False                   MAP -0.01 (0.43 -> 0.42)
  tfidf use_idf=False                     MAP -0.08 (0.43 -> 0.35)
  mode = "bm25" instead of "tfidf"        MAP +0.01 (0.43 -> 0.44)
  tfidf fit all df_{trn,dev,test} + exps  MAP -0.00 (0.44 -> 0.44)
  tfidf fit only exps                     MAP -0.00 (0.44 -> 0.44)
  use rank_bm25 (pip)                     MAP -0.08 (0.44 -> 0.36)
  use CountVectorizer instead of tfidf    MAP -0.09 (0.44 -> 0.35)
  use nltk stopwords in preproc none here MAP +0.01 (0.44 -> 0.45)
  """
  q = maybe_concat_texts(df[field_q].tolist())
  e = maybe_concat_texts(df_exp[field_e].tolist())

  vec = {
      "bm25": BM25Vectorizer(),
      "tfidf": feature_extraction.text.TfidfVectorizer(binary=True),
  }[mode]

  vec.fit(q + e)

  # """
  # Q: If we could somehow obtain/predict the tfidf vector
  # for all the explanation sentences concatenated together,
  # what is the upper bound on the MAP score?
  # A: Dev MAP = 0.812
  # """
  # def concat_exp_text(exp_idxs):
  #     def concat_lst(lst):
  #         return " ".join(lst)
  #     return " ".join(df_exp.lemmas.iloc[exp_idxs].apply(concat_lst).tolist())
  # q = df.exp_idxs.apply(concat_exp_text)

  vectors_q = vec.transform(q)
  vectors_e = vec.transform(e)
  matrix_dist = metrics.pairwise.cosine_distances(vectors_q, vectors_e)

  return [np.argsort(distances) for distances in matrix_dist]


def remove_duplicates(lst: list) -> list:
  seen = set()
  new = []
  for item in lst:
    if item not in seen:
      new.append(item)
      seen.add(item)
  return new


def add_missing_idxs(old, idxs_sample):
  """
  Buggy eval script heavily penalizes missing rankings
  But some methods do not produce ranking over all explanation sentences
  So fill in the missing ones with random order
  """
  old = remove_duplicates(old)
  set_old = set(old)
  set_all = set(idxs_sample)
  missing = list(set_all - set_old)
  # print(set_all - set_old)
  # print(set_old - set_all)
  # print(missing)
  # print(len(old))
  # print(set_old == set_all)
  np.random.shuffle(missing)
  new = list(old)
  new.extend(missing)
  assert len(new) == len(set_all), (len(new), len(set_all), len(missing))
  assert all([a == b for a, b in zip(new[:len(old)], old)])
  assert set(new) == set_all
  return new


def format_predict_line(question_id, explanation_uid):
  """
  Format one line for the prediction text line
  Correct format referenced from baseline tfidf script
  """
  return question_id + "\t" + explanation_uid


def remove_combo_suffix(explanation_uid):
  return explanation_uid.split("_")[0]


def deduplicate_combos(ranks, df_exp):
  idx2idx_canon = get_idx2idx_canon(df_exp)

  def process(lst):
    return remove_duplicates([idx2idx_canon[idx] for idx in lst])

  return [process(lst) for lst in ranks]


def ideal_rerank(ranks, df, df_exp, top_n=64):
  """
  Q: What is the upper bound of the MAP score if we could somehow 
  perfectly re-rank the top_n ranked predictions?
  A: Dev MAP = 0.78 (Using mode=="tfidf", top_n=100)
  """
  def get_combo_idxs(_df_exp):
    uid2idxs = {}
    uids = _df_exp.uid.apply(remove_combo_suffix).tolist()
    for _idx, uid in enumerate(uids):
      if uid not in uid2idxs:
        uid2idxs[uid] = []
      uid2idxs[uid].append(_idx)
    return {_idx: uid2idxs[uid] for _idx, uid in enumerate(uids)}

  idxs_gold = df.exp_idxs.tolist()
  lengths_front, lengths_gold = [], []
  assert len(ranks) == len(idxs_gold) == len(df)
  idx2combo_idxs = get_combo_idxs(df_exp)

  for i in range(len(ranks)):
    temp = []
    for idx in idxs_gold[i]:
      temp.extend(idx2combo_idxs[idx])
    idxs_gold[i] = list(set(temp))
    if len(idxs_gold[i]) > 0:  # test data won't have explanations provided
      front = [idx for idx in ranks[i][:top_n] if idx in idxs_gold[i][:top_n]]
      back = [idx for idx in ranks[i] if idx not in front]
      new = front + back
      assert set(new) == set(ranks[i])
      ranks[i] = new
      lengths_front.append(len(front))
      lengths_gold.append(len(idxs_gold[i]))
  if len(lengths_front) > 0:

    def _process(lengths):
      return round(sum(lengths) / len(lengths), 3)

    f = _process(lengths_front)
    g = _process(lengths_gold)
    print(f"\nGold present in rankings | Predicted: {f} | Total: {g}")
  return ranks


def augment_ranks(ranks_a, ranks_b):
  """
  If ranks_a is too short, the rest can be filled in by ranks_b
  """
  assert len(ranks_a) == len(ranks_b)
  augmented = []
  for front, back in zip(ranks_a, ranks_b):
    set_front = set(front)
    back = [r for r in back if r not in set_front]
    augmented.append(front + back)
  return augmented


def pred_emb_nn_ranking(
    df: pd.DataFrame,
    idx: int,
    model=None,
    annoy_index=None,
    df_exp: pd.DataFrame = None,
) -> list:
  inp = [df.embedding.iloc[idx]]
  out = model.predict(inp)
  pred_emb = out[0]
  return annoy_index.get_nns_by_vector(pred_emb, n=len(df_exp))


def get_ranks(
    df: pd.DataFrame,
    df_exp: pd.DataFrame,
    mode: str = "tfidf",
    use_embed: bool = False,
    use_recursive_tfidf: bool = None,
    field_q: str = "lemmas",
    field_e: str = "lemmas",
) -> list:
  """
  Use ranking methods to get predictions
  Format predictions as required for eval script
  """
  # Setup for simple tfidf ranking
  tfidf_ranking = get_tfidf_ranking(df, df_exp, field_q, field_e)
  tfidf_ranking = deduplicate_combos(tfidf_ranking, df_exp)

  # tfidf_ranking = ideal_rerank(tfidf_ranking, df, df_exp, top_n=16)

  if use_recursive_tfidf:
    recurse_ranks = get_recurse_concat_ranks(df, df_exp)
    recurse_ranks = deduplicate_combos(recurse_ranks, df_exp)
    tfidf_ranking = augment_ranks(recurse_ranks, tfidf_ranking)

  if use_embed:
    # Setup for embedding algos
    idx2nns = {
        idx: list(df_exp.nn_exp.iloc[idx])
        for idx in range(len(df_exp))
    }
  else:
    idx2nns = None

  def scores2ranks(_scores):
    return sorted(
        list(_scores.keys()),
        key=lambda _i: _scores[_i],
        reverse=True,
    )

  uids_all = df_exp.uid.apply(remove_combo_suffix)
  orig_exp_set = set(uids_all.tolist())
  orig_exp_idxs = df_exp[df_exp.uid.isin(orig_exp_set)].index.tolist()
  has_missing = False
  ranks = []

  for i in tqdm(range(len(df))):
    if mode == "repeat":
      # Repeating nn algo (23.7)
      nearest = df.nn_exp.iloc[i]
      scores = repeat(nearest, idx2nns)
      ranked_idxs = scores2ranks(scores)

    elif mode == "recurse":
      # Recursive nn algo (0.07)
      nearest = df.nn_exp.iloc[i]
      scores = {}
      recurse(seed=nearest, scores=scores, idx2nns=idx2nns, iteration=0)
      ranked_idxs = scores2ranks(scores)

    elif mode == "simple_nn":
      # Simple nn algo (0.24)
      ranked_idxs = simple_nn_ranking(df, df_exp, i)

    elif mode == "pred_emb":
      # Predict concat answer embedding algo (0.17)
      ranked_idxs = pred_emb_nn_ranking(df, i)

    elif mode == "tfidf":
      # Tfidf/BM25 ranking algo (0.44)
      ranked_idxs = list(tfidf_ranking[i])

    else:
      raise Exception(f"Unknown mode: {mode}")

    if len(ranked_idxs) != len(orig_exp_set):
      if not has_missing:
        # Print once not every time
        print("Filling in missing rankings with random order")
        has_missing = True
      ranked_idxs = add_missing_idxs(ranked_idxs, idxs_sample=orig_exp_idxs)

    ranks.append(ranked_idxs)
  return ranks


def get_preds(ranks: list, df: pd.DataFrame, df_exp: pd.DataFrame) -> list:
  preds = []
  assert len(ranks) == len(df)
  uids = df_exp.uid.apply(remove_combo_suffix).values
  qids = df.questionID.tolist()
  for i in range(len(df)):
    uids_pred = uids[ranks[i]]
    preds.extend([format_predict_line(qids[i], uid) for uid in uids_pred])
  return preds


def write_preds(preds: list, path: str = "predict.txt") -> None:
  """
  Write prediction text file for eval script
  Preds must be already formatted
  """
  with open(path, "w") as f:
    f.write("\n".join(preds))


def test_write_preds():
  preds = [
      "VASoL_2008_3_26\t14de-6699-6b2e-a5d1",
      "VASoL_2008_3_26\t14de-6699-6b2e-a5d1",
  ]
  path = "temp.txt"
  write_preds(preds, path)
  with open(path) as f:
    for line in f:
      print(repr(line))


def run_scoring(path_gold: Path, path_predict: str = "predict.txt") -> dict:
  """
  Eval function with callback to get score for each question
  Uses ranking/scoring implementation from original script
  """
  gold = evaluate.load_gold(str(path_gold))
  pred = evaluate.load_pred(str(path_predict))

  qid2score = {}

  def _callback(qid, score):
    qid2score[qid] = score

  mean_ap = evaluate.mean_average_precision_score(
      gold,
      pred,
      callback=_callback,
  )
  print("qid2score:", qid2score)
  print("MAP: ", mean_ap)
  return qid2score
