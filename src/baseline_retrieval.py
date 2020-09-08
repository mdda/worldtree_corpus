import pandas as pd
import json
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict

import numpy as np
from fire import Fire
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

from bm25 import BM25Vectorizer
from dataset import Statement, QuestionAnswer, TxtAndKeywords
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
    # Dev MAP = 0.37
    vectorizer: TfidfVectorizer = BM25Vectorizer()

    class Config:
        arbitrary_types_allowed = True

    def make_pred(self, i_query: int, rank: List[int], data: Data) -> Prediction:
        uids = [data.statements[i].uid_base for i in rank]
        uids = deduplicate(uids)
        return Prediction(qid=data.questions[i_query].question_id, uids=uids)

    def preprocess(self, x: Union[TxtAndKeywords, Statement]) -> str:
        return x.raw_txt

    def rank(self, vecs_q: np.ndarray, vecs_s: np.ndarray) -> np.ndarray:
        distances: np.ndarray = cosine_distances(vecs_q, vecs_s)
        ranking: np.ndarray = np.argsort(distances, axis=-1)
        return ranking

    def run(self, data: Data) -> List[Prediction]:
        statements: List[str] = [self.preprocess(s) for s in data.statements]
        queries: List[str] = [
            self.preprocess(q.question) + " " + self.preprocess(q.answers[0])
            for q in data.questions
        ]
        self.vectorizer.fit(statements + queries)
        ranking = self.rank(
            self.vectorizer.transform(queries), self.vectorizer.transform(statements)
        )
        preds: List[Prediction] = []
        for i in tqdm(range(len(ranking))):
            preds.append(self.make_pred(i, list(ranking[i]), data))
        return preds


class StageRanker(SimpleRanker):
    # Dev MAP: 0.3816
    num_per_stage: List[int] = [25, 100]

    def recurse(
        self,
        vec_q: np.ndarray,
        vecs_s: np.ndarray,
        indices_s: np.ndarray,
        num_per_stage: List[int],
    ) -> List[int]:
        num_s = vecs_s.shape[0]
        assert num_s == len(indices_s)
        if num_s == 0:
            return []

        num_keep = num_s
        if num_per_stage:
            num_keep = num_per_stage.pop(0)
        num_next = max(num_s - num_keep, 0)

        distances: np.ndarray = cosine_distances(vec_q, vecs_s)[0]
        assert distances.shape == (num_s,)
        rank = np.argsort(distances)

        vecs_keep = vecs_s[rank][:num_keep]
        indices_keep = indices_s[rank][:num_keep]
        vec_q = np.max(vecs_keep, axis=0)

        if num_next == 0:
            vecs_s = np.array([])
            indices_s = np.array([])
        else:
            vecs_s = vecs_s[rank][-num_next:]
            indices_s = indices_s[rank][-num_next:]

        return list(indices_keep) + self.recurse(
            vec_q, vecs_s, indices_s, num_per_stage
        )

    def rank(self, vecs_q: np.ndarray, vecs_s: np.ndarray) -> np.ndarray:
        num_q = vecs_q.shape[0]
        num_s = vecs_s.shape[0]
        ranking = np.zeros(shape=(num_q, num_s), dtype=np.int)
        for i in tqdm(range(num_q)):
            ranking[i] = self.recurse(
                vecs_q[i], vecs_s, np.arange(num_s), list(self.num_per_stage)
            )
        return ranking


class TextGraphsLemmatizer(BaseModel):
    root: str = "/tmp/TextGraphLemmatizer"
    url: str = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/worldtree_textgraphs_2019_010920.zip"
    sep: str = "\t"
    word_to_lemma: Optional[Dict[str, str]]

    def read_csv(
        self, path: Path, header: str = None, names: List[str] = None
    ) -> pd.DataFrame:
        return pd.read_csv(path, header=header, names=names, sep=self.sep)

    @staticmethod
    def preprocess(word: str) -> str:
        # Remove punct eg dry-clean -> dryclean so
        # they won't get split by downstream tokenizers
        word = word.lower()
        word = "".join([c for c in word if c.isalpha()])
        return word

    def load(self):
        if not self.word_to_lemma:
            download_and_extract_archive(self.url, self.root, self.root)
            path = list(Path(self.root).glob("**/annotation"))[0]
            df = self.read_csv(path / "lemmatization-en.txt", names=["lemma", "word"])
            self.word_to_lemma = {}
            for word, lemma in df.values:
                self.word_to_lemma[self.preprocess(word)] = self.preprocess(lemma)

    def run(self, text: str) -> str:
        self.load()
        return " ".join(self.word_to_lemma.get(w, w) for w in text.split())

#import spacy
#nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # "tagger", "parser", 

class LemmaRanker(SimpleRanker):
    # Dev MAP:  0.3640
    lemmatizer = TextGraphsLemmatizer()

    def preprocess(self, x: Union[TxtAndKeywords, Statement]) -> str:
        #print(x.raw_txt)
        if False: # Scores 0.4370 on dev
            # Needs import spacy and spacy.load above
            doc = nlp(x.raw_txt)
            words = [token.lemma_ for token in doc]
            #words = sorted(list(set(words))) # No change
        if True:  # Scores 0.4311 on dev
            words = x.keywords

        # return self.lemmatizer.run(x.raw_txt)
        #words = x.keywords
        #words = ["".join([c for c in w if c.isalnum()]) for w in words]
        return " ".join(words)


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
    # ranker = SimpleRanker()
    # ranker = StageRanker()
    ranker = LemmaRanker()
    preds = ranker.run(data)
    Scorer().run(data.path_gold, preds)


if __name__ == "__main__":
    Fire(main)
