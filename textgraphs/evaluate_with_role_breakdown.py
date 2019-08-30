# From https://github.com/umanlp/tg2019task/blob/master/evaluate.py

import math
import sys
import warnings
from collections import namedtuple, OrderedDict
from functools import partial

import pandas as pd


class ListShouldBeEmptyWarning(UserWarning):
  pass


Question = namedtuple('Question', 'id explanations')
Explanation = namedtuple('Explanation', 'id role')


def format_role(role: str) -> str:
  if role == "NE":
    return "NEG"
  return role


def load_gold(filepath_or_buffer, sep='\t'):
  df = pd.read_csv(filepath_or_buffer, sep=sep)

  gold = OrderedDict()

  for _, row in df[['questionID', 'explanation']].dropna().iterrows():
    # explanations = OrderedDict((uid.lower(), Explanation(uid.lower(), role))
    explanations = OrderedDict(
        (uid.lower(), Explanation(uid.lower(), format_role(role)))
        for e in row['explanation'].split()
        for uid, role in (e.split('|', 1), ))

    question = Question(row['questionID'].lower(), explanations)

    gold[question.id] = question

  return gold


def load_pred(filepath_or_buffer, sep='\t'):
  df = pd.read_csv(filepath_or_buffer,
                   sep=sep,
                   names=('question', 'explanation'))

  pred = OrderedDict()

  for question_id, df_explanations in df.groupby('question'):
    pred[question_id.lower()] = list(
        OrderedDict.fromkeys(df_explanations['explanation'].str.lower()))

  return pred


def compute_ranks(true, pred):
  ranks = []

  if not true or not pred:
    return ranks

  targets = list(true)

  # I do not understand the corresponding block of the original Scala code.
  for i, pred_id in enumerate(pred):
    for true_id in targets:
      if pred_id == true_id:
        ranks.append(i + 1)
        targets.remove(pred_id)
        break

  # Example: Mercury_SC_416133
  if targets:
    warnings.warn(
        'targets list should be empty, but it contains: ' + ', '.join(targets),
        ListShouldBeEmptyWarning)

    for _ in targets:
      ranks.append(0)

  return ranks


def average_precision(ranks):
  total = 0.

  if not ranks:
    return total

  for i, rank in enumerate(ranks):
    precision = float(i + 1) / float(rank) if rank > 0 else math.inf
    total += precision

  return total / len(ranks)


def per_role_map_score(gold: OrderedDict, pred: OrderedDict) -> None:
  role_counts = {}
  for question in gold.values():
    for explanation in question.explanations.values():
      role = explanation.role
      if role not in role_counts:
        role_counts[role] = 0
      role_counts[role] += 1
  roles = sorted(role_counts.keys())

  def process(_role: str) -> (float, float, float):
    total, count = 0., 0
    for _question in gold.values():
      exp_ids = [
          exp_id for exp_id in list(_question.explanations)
          if _question.explanations[exp_id].role == _role
      ]
      # print("Role:", role, "num found:", len(exp_ids))
      ranks = compute_ranks(exp_ids, pred[_question.id])

      score = average_precision(ranks)

      if not math.isfinite(score):
        score = 0.

      total += score
      count += 1

    mean_ap = total / count if count > 0 else 0.
    return mean_ap, total, count

  data = []
  for role in roles:
    _mean_ap, _total, _count = process(role)
    data.append([role, _mean_ap, _total, _count, role_counts[role]])
  cols = ["role", "MAP", "total", "count", "occurences"]
  assert len(cols) == len(data[0])
  df = pd.DataFrame(data, columns=cols)
  with pd.option_context("display.max_columns", None):
    print(df.round(4))


def mean_average_precision_score(gold, pred, callback=None):
  per_role_map_score(gold, pred)

  total, count = 0., 0

  for question in gold.values():
    if question.id in pred:
      ranks = compute_ranks(list(question.explanations), pred[question.id])

      score = average_precision(ranks)

      if not math.isfinite(score):
        score = 0.

      total += score
      count += 1

      if callback:
        callback(question.id, score)

  mean_ap = total / count if count > 0 else 0.

  return mean_ap


def main():
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--gold',
                      type=argparse.FileType('r', encoding='UTF-8'),
                      required=True)
  parser.add_argument('pred', type=argparse.FileType('r', encoding='UTF-8'))
  args = parser.parse_args()

  gold, pred = load_gold(args.gold), load_pred(args.pred)

  # callback is optional, here it is used to print intermediate results to STDERR
  mean_ap = mean_average_precision_score(gold,
                                         pred,
                                         callback=partial(print,
                                                          file=sys.stderr))

  print('MAP: ', mean_ap)


if '__main__' == __name__:
  main()
