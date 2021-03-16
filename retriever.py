from collections import defaultdict
import pickle

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


class PredictManager:
    @staticmethod
    def write(path, preds):
        lines = []
        for pred in preds:
            for explanation in pred.eids:
                lines.append("\t".join([pred.qid, explanation]))
        with open(path, "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def read(path):
        qid_preds = defaultdict(list)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                qid, eid = line.split("\t")
                qid_preds[qid].append(eid)
        preds = [Prediction(qid=qid, eids=eids) for qid, eids in qid_preds.items()]
        return preds

    @staticmethod
    def dump(path, preds):
        with open(path, "wb") as f:
            pickle.dump(preds, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
