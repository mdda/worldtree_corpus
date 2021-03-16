from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ranker import CosineRanker


class Prediction:
    def __init__(self, qid, eids):
        self.qid = qid
        self.eids = eids

    def __eq__(self, other):
        if self.qid != other.qid:
            return False
        for e1, e2 in zip(self.eids, other.eids):
            if e1 != e2:
                return False

        return True


class Retriever:
    vectorizer = TfidfVectorizer()
    ranker = CosineRanker()

    def __init__(self, limit=100):
        self.limit = limit

    def run(self, questions, explanations):
        ranking = self.rank(
            questions["question_text"], explanations["explanation_text"]
        )
        preds = []
        for i in tqdm(range(len(ranking))):
            preds.append(
                Prediction(
                    qid=questions.loc[i]["question_id"],
                    eids=[
                        explanations.loc[j]["explanation_id"]
                        for j in ranking[i][: self.limit]
                    ],
                )
            )
        return preds

    def rank(self, questions, explanations):
        self.vectorizer.fit(questions).fit(explanations)
        return self.ranker.run(
            self.vectorizer.transform(questions),
            self.vectorizer.transform(explanations),
        )
