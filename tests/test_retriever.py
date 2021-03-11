import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from retriever import Retriever
from dataset import QuestionRatingDataset, ExplanationDataset


def test_retriever():
    questions = QuestionRatingDataset("data/wt-expert-ratings.dev.json").questions
    explanations = ExplanationDataset("data/tables").explanations

    limit = 10
    retriever = Retriever(limit=10)
    actual_preds = retriever.run(questions, explanations)

    vectorizer = (
        TfidfVectorizer()
        .fit(questions["question_text"])
        .fit(explanations["explanation_text"])
    )
    X_q = vectorizer.transform(questions["question_text"])
    X_e = vectorizer.transform(explanations["explanation_text"])
    X_dist = cosine_distances(X_q, X_e)

    expected_preds = []
    for i_question, distances in enumerate(X_dist):
        eids = []
        for i_explanation in np.argsort(distances)[:limit]:
            eids.append(explanations.loc[i_explanation]["explanation_id"])
        expected_preds.append(
            {"qid": questions.loc[i_question]["question_id"], "eids": eids}
        )
    assert actual_preds == expected_preds
