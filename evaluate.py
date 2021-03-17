from typing import Dict, List, Iterable

from tqdm import tqdm
import numpy as np

from retriever import PredictManager
from dataset import QuestionRatingDataset


def mean_average_ndcg(gold_preds, preds, rating_threshold):
    scores = []
    oracle_scores = []
    for pred in tqdm(preds):
        qid = pred.qid
        score = ndcg(gold_preds[qid], pred.eids, rating_threshold)
        oracle_score = ndcg(gold_preds[qid], pred.eids, rating_threshold, oracle=True)
        scores.append(score)
        oracle_scores.append(oracle_score)
    print(f"ndcg: {np.average(scores)}")
    print(f"oracle ndcg: {np.average(oracle_scores)}")


def ndcg(
    gold: Dict[str, float],
    predicted: List[str],
    rating_threshold: int,
    alternate: bool = True,
    oracle: bool = False,
) -> float:
    """Calculate NDCG value for individual Question-Explanations Pair

    Args:
        gold (Dict[str, float]): Gold expert ratings
        predicted (List[str]): List of predicted ids
        rating_threshold (int): Threshold of gold ratings to consider for NDCG calcuation
        alternate (bool, optional): True to use the alternate scoring (intended to place more emphasis on relevant results). Defaults to True.

    Returns:
        float: NDCG score
    """
    if len(gold) == 0:
        return 1

    # Only consider relevance scores greater than 2
    relevance = np.array(
        [
            gold[f_id] if f_id in gold and gold[f_id] > rating_threshold else 0
            for f_id in predicted
        ]
    )
    if oracle:
        relevance = sorted(relevance)[::-1]

    missing_ids = [g_id for g_id in gold if g_id not in predicted]

    if len(missing_ids) > 0:
        padded = np.zeros(10 ** 6)
        for index, g_id in enumerate(missing_ids):
            padded[index] = gold[g_id]
        relevance = np.concatenate((relevance, np.flip(padded)), axis=0)

    nranks = len(relevance)

    if relevance is None or len(relevance) < 1:
        return 0.0

    if nranks < 1:
        raise Exception("nranks < 1")

    pad = max(0, nranks - len(relevance))

    # pad could be zero in which case this will no-op
    relevance = np.pad(relevance, (0, pad), "constant")

    # now slice downto nranks
    relevance = relevance[0 : min(nranks, len(relevance))]

    ideal_dcg = idcg(relevance, alternate)
    if ideal_dcg == 0:
        return 0.0

    return dcg(relevance, alternate) / ideal_dcg


def dcg(relevance: np.array, alternate: bool = True) -> float:
    """Calculate discounted cumulative gain.

    Args:
        relevance (np.array): Graded and ordered relevances of the results.
        alternate (bool, optional): True to use the alternate scoring (intended to place more emphasis on relevant results). Defaults to True.

    Returns:
        float: DCG score
    """
    if relevance is None or len(relevance) < 1:
        return 0.0

    p = len(relevance)
    if alternate:
        # from wikipedia: "An alternative formulation of
        # DCG[5] places stronger emphasis on retrieving relevant documents"

        log2i = np.log2(np.asarray(range(1, p + 1)) + 1)
        return ((np.power(2, relevance) - 1) / log2i).sum()
    else:
        log2i = np.log2(range(2, p + 1))
        return relevance[0] + (relevance[1:] / log2i).sum()


def idcg(relevance: np.array, alternate: bool = True) -> float:
    """Calculate ideal discounted cumulative gain (maximum possible DCG)

    Args:
        relevance (np.array): Graded and ordered relevances of the results
        alternate (bool, optional): True to use the alternate scoring (intended to place more emphasis on relevant results).. Defaults to True.

    Returns:
        float: IDCG Score
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    # guard copy before sort
    rel = relevance.copy()
    rel.sort()
    return dcg(rel[::-1], alternate)


if __name__ == "__main__":
    dataset = QuestionRatingDataset("data/wt-expert-ratings.dev.json")
    ge = dataset.gold_predictions
    preds = PredictManager.read("predict.dev.model.txt")
    mean_average_ndcg(ge, preds, 0)
