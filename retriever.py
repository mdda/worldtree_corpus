from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ranker import CosineRanker


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
                {
                    "qid": questions.loc[i]["question_id"],
                    "eids": [
                        explanations.loc[j]["explanation_id"]
                        for j in ranking[i][: self.limit]
                    ],
                }
            )
        return preds

    def rank(self, questions, explanations):
        self.vectorizer.fit(questions).fit(explanations)
        return self.ranker.run(
            self.vectorizer.transform(questions),
            self.vectorizer.transform(explanations),
        )
