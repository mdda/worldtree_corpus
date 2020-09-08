import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from fire import Fire
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from bm25 import BM25Vectorizer
from dataset import Statement, QuestionAnswer
from extra_data import SplitEnum

sys.path.append("../tg2020task")

import evaluate


def deduplicate(items: list) -> list:
    seen = set()
    output = []
    for x in items:
        if x not in seen:
            seen.add(x)
            output.append(x)
    return output


class Data(BaseModel):
    root: str = "../data"
    root_gold: str = "../tg2020task"
    data_split: str = SplitEnum.dev
    statements: Optional[List[Statement]]
    questions: Optional[List[QuestionAnswer]]

    @property
    def path_gold(self) -> Path:
        return Path(self.root_gold) / f"questions.{self.data_split}.tsv"

    @staticmethod
    def load_jsonl(path: Path) -> List[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                assert line
                records.append(json.loads(line))
        return records

    def load(self):
        path = Path(self.root) / "statements.jsonl"
        self.statements = [Statement(**r) for r in self.load_jsonl(path)]
        path = Path(self.root) / f"questions.{self.data_split}.jsonl"
        self.questions = [QuestionAnswer(**r) for r in self.load_jsonl(path)]

    def analyze(self):
        info = dict(statements=len(self.statements), questions=len(self.questions))
        print(info)


class Prediction(BaseModel):
    qid: str
    uids: List[str]
    scores: Optional[List[float]]
    sep: str = "\t"

    @property
    def lines(self) -> List[str]:
        return [self.sep.join([self.qid, p]) for p in self.uids]


class SimpleRanker(BaseModel):
    vectorizer: TfidfVectorizer = BM25Vectorizer()

    class Config:
        arbitrary_types_allowed = True

    def make_pred(self, i_query: int, rank: List[int], data: Data) -> Prediction:
        uids = [data.statements[i].uid_base for i in rank]
        uids = deduplicate(uids)
        return Prediction(qid=data.questions[i_query].question_id, uids=uids)

    def run(self, data: Data) -> List[Prediction]:
        statements: List[str] = [s.raw_txt for s in data.statements]
        queries: List[str] = [
            q.question.raw_txt + " " + q.answers[0].raw_txt for q in data.questions
        ]
        self.vectorizer.fit(statements + queries)
        distances: np.ndarray = cosine_distances(
            self.vectorizer.transform(queries), self.vectorizer.transform(statements)
        )
        ranking: np.ndarray = np.argsort(distances, axis=-1)
        assert ranking.shape == (len(queries), len(statements))

        preds: List[Prediction] = []
        for i in tqdm(range(len(ranking))):
            preds.append(self.make_pred(i, list(ranking[i]), data))
        return preds


class Scorer(BaseModel):
    root: str = "/tmp/scorer"
    sep: str = "\n"

    def run(self, path_gold: Path, preds: List[Prediction]):
        lines = [x for p in preds for x in p.lines]
        path_predict = Path(self.root) / "predict.txt"
        path_predict.parent.mkdir(exist_ok=True)
        with open(path_predict, "w") as f:
            f.write(self.sep.join(lines))

        gold = evaluate.load_gold(str(path_gold))
        pred = evaluate.load_pred(str(path_predict))

        qid2score = {}

        def _callback(qid, score):
            qid2score[qid] = score

        mean_ap = evaluate.mean_average_precision_score(gold, pred, callback=_callback)
        # print("qid2score:", qid2score)
        print("MAP: ", mean_ap)
        return qid2score


def main(data_split=SplitEnum.dev):
    data = Data(data_split=data_split)
    data.load()
    data.analyze()
    ranker = SimpleRanker()
    preds = ranker.run(data)
    Scorer().run(data.path_gold, preds)


if __name__ == "__main__":
    Fire(main)
