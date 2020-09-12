import json
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
import spacy
from fire import Fire
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import Pipeline
from spacy.lang.en import English
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

from bm25 import BM25Vectorizer
from dataset import (
    Statement,
    QuestionAnswer,
    TxtAndKeywords,
    load_qanda,
    load_statements,
)
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
    uid_to_statements: Optional[Dict[str, List[Statement]]]

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
        self.statements = load_statements()
        self.questions = load_qanda(self.data_split)

        self.uid_to_statements = {}
        for s in self.statements:
            self.uid_to_statements.setdefault(s.uid_base, []).append(s)

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


class TextProcessor(BaseModel):
    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        return x.raw_txt


class Ranker(BaseModel):
    def run(self, vecs_q: csr_matrix, vecs_s: csr_matrix) -> np.ndarray:
        distances: np.ndarray = cosine_distances(vecs_q, vecs_s)
        ranking: np.ndarray = np.argsort(distances, axis=-1)
        return ranking


class Retriever(BaseModel):
    preproc: TextProcessor = TextProcessor()
    vectorizer: Union[TfidfVectorizer, Pipeline] = BM25Vectorizer()
    ranker: Ranker = Ranker()

    class Config:
        arbitrary_types_allowed = True

    def make_pred(self, i_query: int, rank: List[int], data: Data) -> Prediction:
        uids = [data.statements[i].uid_base for i in rank]
        uids = deduplicate(uids)
        return Prediction(qid=data.questions[i_query].question_id, uids=uids)

    def make_query(self, q: QuestionAnswer, data: Data) -> str:
        assert data
        return self.preproc.run(q.question) + " " + self.preproc.run(q.answers[0])

    def rank(self, queries: List[str], statements: List[str]) -> np.ndarray:
        self.vectorizer.fit(statements + queries)
        return self.ranker.run(
            self.vectorizer.transform(queries), self.vectorizer.transform(statements)
        )

    def run(self, data: Data) -> List[Prediction]:
        statements: List[str] = [self.preproc.run(s) for s in data.statements]
        queries: List[str] = [self.make_query(q, data) for q in data.questions]
        ranking = self.rank(queries, statements)
        preds: List[Prediction] = []
        for i in tqdm(range(len(ranking))):
            preds.append(self.make_pred(i, list(ranking[i]), data))
        return preds


class StageRanker(Ranker):
    # Dev MAP: 0.3816
    num_per_stage: List[int] = [25, 100]
    scale: float = 1.0

    def recurse(
        self,
        vec_q: csr_matrix,
        vecs_s: csr_matrix,
        indices_s: np.ndarray,
        num_per_stage: List[int],
        num_accum: int = 0,
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
        vec_new = np.max(vecs_keep, axis=0)
        vec_new = vec_new / (self.scale ** num_accum)
        if isinstance(vec_q, csr_matrix):
            vec_q = vec_q.maximum(vec_new)
        else:
            vec_q = np.maximum(vec_q, vec_new)
        num_accum += num_keep

        if num_next == 0:
            vecs_s = np.array([])
            indices_s = np.array([])
        else:
            vecs_s = vecs_s[rank][-num_next:]
            indices_s = indices_s[rank][-num_next:]

        return list(indices_keep) + self.recurse(
            vec_q, vecs_s, indices_s, num_per_stage, num_accum,
        )

    def run(self, vecs_q: csr_matrix, vecs_s: csr_matrix) -> np.ndarray:
        num_q = vecs_q.shape[0]
        num_s = vecs_s.shape[0]
        ranking = np.zeros(shape=(num_q, num_s), dtype=np.int)
        for i in tqdm(range(num_q)):
            ranking[i] = self.recurse(
                vecs_q[[i]], vecs_s, np.arange(num_s), list(self.num_per_stage)
            )
        return ranking


class TextGraphsLemmatizer(TextProcessor):
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

    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        self.load()
        text = x.raw_txt
        return " ".join(self.word_to_lemma.get(w, w) for w in text.split())


class SpacyProcessor(TextProcessor):
    nlp: English = spacy.load("en_core_web_sm", disable=["tagger", "ner", "parser"])

    class Config:
        arbitrary_types_allowed = True

    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        doc = self.nlp(x.raw_txt)
        words = [token.lemma_ for token in doc]
        return " ".join(words)


class KeywordProcessor(TextProcessor):
    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        return " ".join(x.keywords)


class IterativeRanker(Ranker):
    max_seq_len: int = 128
    top_n: int = 1
    scale: float = 1.25

    def recurse(
        self, vec_q: csr_matrix, vecs_s: csr_matrix, indices: List[int],
    ):
        if len(indices) >= self.max_seq_len:
            return

        distances: np.ndarray = cosine_distances(vec_q, vecs_s)[0]
        assert distances.shape == (vecs_s.shape[0],)
        rank = np.argsort(distances)

        seen = set(indices)
        count = 0
        for i in rank:
            if count == self.top_n:
                break

            if i not in seen:
                vec_new = vecs_s[i] / (self.scale ** len(indices))
                if isinstance(vec_q, csr_matrix):
                    vec_q = vec_q.maximum(vec_new)
                else:
                    vec_q = np.maximum(vec_q, vec_new)
                indices.append(i)
                count += 1
                self.recurse(vec_q, vecs_s, indices)

    def run(self, vecs_q: csr_matrix, vecs_s: csr_matrix) -> np.ndarray:
        ranking = super().run(vecs_q, vecs_s)
        rank_old: np.ndarray
        for i, rank_old in tqdm(enumerate(ranking), total=len(ranking)):
            rank_new = []
            self.recurse(vecs_q[[i]], vecs_s, rank_new)
            assert rank_new
            rank_new = rank_new + list(rank_old)
            rank_new = deduplicate(rank_new)
            ranking[i] = rank_new
        return ranking


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
        # print(len(gold), len(pred))  # 410 496

        qid2score = {}

        def _callback(qid, score):
            qid2score[qid] = score

        mean_ap = evaluate.mean_average_precision_score(gold, pred, callback=_callback)
        # print("qid2score:", qid2score)
        print("MAP: ", mean_ap)

        with open("/tmp/scorer/per_q.json", "wt") as f:
            json.dump(qid2score, f)

        return qid2score


class ResultAnalyzer(BaseModel):
    thresholds: List[int] = [32, 64, 128, 256, 512, 1024]

    @staticmethod
    def run_threshold(data: Data, preds: List[Prediction], threshold: int) -> float:
        assert len(data.questions) == len(preds)
        scores = []

        for q, p in zip(data.questions, preds):
            assert q.question_id == p.qid
            predicted = set(p.uids[:threshold])
            gold = set([e.uid for e in q.explanation_gold])
            if gold:
                true_pos = predicted.intersection(gold)
                recall = len(true_pos) / len(gold)
                scores.append(recall)
            else:
                print(dict(question_no_labels=q.question_id))

        return sum(scores) / len(scores)

    def run(self, data: Data, preds: List[Prediction]):
        for threshold in self.thresholds:
            recall = self.run_threshold(data, preds, threshold)
            print(dict(threshold=threshold, recall=recall))


class TruncatedSVDVectorizer(TfidfVectorizer):
    def __init__(self, vec: TfidfVectorizer, n_components: int, random_state=42):
        super().__init__()
        self.vec = vec
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    def fit(self, texts: List[str], y=None):
        self.vec.fit(texts)
        x = self.vec.transform(texts)
        self.svd.fit(x)

    def transform(self, texts: List[str], copy="deprecated"):
        x = self.vec.transform(texts)
        x = self.svd.transform(x)
        return x


def main(data_split=SplitEnum.dev):
    data = Data(data_split=data_split)
    data.load()
    data.analyze()

    # retriever = Retriever()  # Dev MAP:  0.3788
    # retriever = Retriever(preproc=SpacyProcessor())  # Dev MAP:  0.4378
    # retriever = Retriever(preproc=KeywordProcessor())  # Dev MAP:  0.4311
    # retriever = Retriever(
    #     preproc=KeywordProcessor(), ranker=StageRanker()
    # )  # Dev MAP:  0.4354, Dev recall@512=0.9084
    if False:
        retriever = Retriever(
            preproc=KeywordProcessor(), ranker=IterativeRanker()
        )  # Dev MAP:  0.4505, Dev recall@512=0.8880
    if True:
        retriever = Retriever(
            preproc=KeywordProcessor(),
            ranker=StageRanker(num_per_stage=[16, 32, 64, 128], scale=1.5),
        )  # Dev MAP:  0.4368, Dev recall@512=0.9177

        # Maybe this dense vectorizer can make useful features for deep learning methods
        # retriever = Retriever(
        #     vectorizer=TruncatedSVDVectorizer(BM25Vectorizer(), n_components=768),
        #     preproc=KeywordProcessor(),
        # )  # Dev MAP:  0.3596, Dev recall@512=0.8684

    preds = retriever.run(data)
    Scorer().run(data.path_gold, preds)
    ResultAnalyzer().run(data, preds)


if __name__ == "__main__":
    Fire(main)
