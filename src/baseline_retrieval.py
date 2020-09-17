import json
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
from fire import Fire
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import English
from spacy.tokens import Token
from torch import nn
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

from dataset import (
    Statement,
    QuestionAnswer,
    TxtAndKeywords,
    load_qanda,
    load_statements,
)
from extra_data import SplitEnum
from losses import APLoss
from rankers import Ranker, StageRanker, IterativeRanker, deduplicate
from vectorizers import BM25Vectorizer, TruncatedSVDVectorizer

sys.path.append("../tg2020task")
import evaluate


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


class Retriever(BaseModel):
    preproc: TextProcessor = TextProcessor()
    vectorizer: TfidfVectorizer = BM25Vectorizer()
    ranker: Ranker = Ranker()

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def make_pred(i_query: int, rank: List[int], data: Data) -> Prediction:
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
    remove_stopwords: bool = False

    class Config:
        arbitrary_types_allowed = True

    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        doc = self.nlp(x.raw_txt)
        tokens: List[Token] = [tok for tok in doc]
        if self.remove_stopwords:
            # Only 1 case where all tokens are stops: "three is more than two"
            tokens = [tok for tok in tokens if not tok.is_stop]
            if not tokens:
                print("SpacyProcessor: No non-stopwords:", doc)
                return "nothing"
        words = [tok.lemma_ for tok in tokens]
        return " ".join(words)


class KeywordProcessor(TextProcessor):
    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        return " ".join(x.keywords)


def write_preds(preds: List[Prediction], path: str, sep="\n"):
    lines = [x for p in preds for x in p.lines]
    with open(path, "w") as f:
        f.write(sep.join(lines))
    # assert [p.dict() for p in preds] == [p.dict() for p in read_preds(path)]


def read_preds(path: str, sep="\t") -> List[Prediction]:
    qid_to_uids = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            qid, uid = line.split(sep)
            qid_to_uids.setdefault(qid, []).append(uid)
    preds = [Prediction(qid=qid, uids=uids) for qid, uids in qid_to_uids.items()]
    return preds


class Scorer(BaseModel):
    @staticmethod
    def run(path_gold: Path, path_predict: Path):
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
    thresholds: List[int] = [64, 128, 256, 512]
    loss_fn: nn.Module = APLoss()

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def filter_qns(
        data: Data, preds: List[Prediction]
    ) -> Tuple[List[QuestionAnswer], List[Prediction]]:
        assert len(data.questions) == len(preds)
        pairs = []
        for q, p in zip(data.questions, preds):
            if q.explanation_gold:
                pairs.append((q, p))
            else:
                print(dict(qn_no_explains=q.question_id))

        qns, preds = zip(*pairs)
        return qns, preds

    def run_threshold(
        self, data: Data, preds: List[Prediction], threshold: int
    ) -> float:
        qns, preds = self.filter_qns(data, preds)
        scores = []

        for q, p in zip(qns, preds):
            assert q.question_id == p.qid
            predicted = set(p.uids[:threshold])
            gold = set([e.uid for e in q.explanation_gold])
            if gold:
                true_pos = predicted.intersection(gold)
                recall = len(true_pos) / len(gold)
                scores.append(recall)

        return sum(scores) / len(scores)

    def run_map_loss(
        self, data: Data, preds: List[Prediction], top_n: int = None
    ) -> float:
        qns, preds = self.filter_qns(data, preds)
        if top_n is None:
            top_n = len(preds[0].uids)
        losses = []

        scores = torch.div(1.0, torch.arange(top_n) + 1)  # [1, 0.5, 0.3  ... 0]
        scores = torch.unsqueeze(scores, dim=0)

        for i, q in enumerate(qns):
            uids = preds[i].uids[:top_n]
            uids_gold = set([e.uid for e in q.explanation_gold])
            labels = [int(u in uids_gold) for u in uids]
            if sum(labels) == 0:
                continue
            loss = self.loss_fn(scores, torch.Tensor(labels).unsqueeze(dim=0))
            losses.append(loss.item())
        return sum(losses) / len(losses)

    @staticmethod
    def count_qns_no_hits(data: Data, preds: List[Prediction], top_n: int) -> int:
        count = 0
        for q, p in zip(data.questions, preds):
            uids_gold = set([e.uid for e in q.explanation_gold])
            if not uids_gold.intersection(p.uids[:top_n]):
                count += 1
        return count

    def run(self, data: Data, preds: List[Prediction]):
        records = []
        for threshold in self.thresholds + [len(preds[0].uids)]:
            recall = self.run_threshold(data, preds, threshold)
            qns_no_hits = self.count_qns_no_hits(data, preds, threshold)
            map_loss = self.run_map_loss(data, preds, threshold)
            records.append(
                dict(
                    threshold=threshold,
                    recall=recall,
                    qns_no_hits=qns_no_hits,
                    map_loss=map_loss,
                )
            )
        df = pd.DataFrame(records)
        print(df)


def main(
    data_split=SplitEnum.dev,
    output_pattern="../predictions/predict.FOLD.baseline-retrieval.txt",
):
    data = Data(data_split=data_split)
    data.load()
    data.analyze()
    path_predict = output_pattern.replace("FOLD", data_split)

    retrievers = [
        Retriever(),  # Dev MAP=0.3965, recall@512=0.7911
        Retriever(preproc=SpacyProcessor()),  # Dev MAP=0.4587, recall@512=0.8849
        Retriever(
            preproc=SpacyProcessor(remove_stopwords=True)
        ),  # Dev MAP=0.4615, recall@512=0.8780
        Retriever(preproc=KeywordProcessor()),  # Dev MAP=0.4529, recall@512=0.8755
        Retriever(
            preproc=KeywordProcessor(), ranker=StageRanker()
        ),  # Dev MAP=0.4575, recall@512=0.9095
        # Maybe this dense vectorizer can make useful features for deep learning methods
        Retriever(
            vectorizer=TruncatedSVDVectorizer(BM25Vectorizer(), n_components=768),
            preproc=KeywordProcessor(),
        ),  # Dev MAP:  0.3741, recall@512= 0.8772
        Retriever(
            preproc=KeywordProcessor(), ranker=IterativeRanker()
        ),  # Dev MAP=0.4704, recall@512=0.8910
        Retriever(
            preproc=KeywordProcessor(),
            ranker=StageRanker(num_per_stage=[16, 32, 64, 128], scale=1.5),
        ),  # Dev MAP=0.4586, recall@512=0.9242
        # Loading SpacyVectorizer has an annoying delay
        # Retriever(
        #     preproc=SpacyProcessor(remove_stopwords=True),
        #     vectorizer=SpacyVectorizer(),
        #     ranker=WordEmbedRanker(),
        # ),  # Dev MAP=0.01436, recall@512=0.4786
        # From hyperopt_retrieval.py
        Retriever(
            preproc=SpacyProcessor(remove_stopwords=True),
            ranker=StageRanker(num_per_stage=[1, 2, 4, 8, 16], scale=1.25),
            vectorizer=BM25Vectorizer(binary=True, use_idf=True, k1=2.0, b=0.5),
        ),  # Dev MAP=0.4861, recall@512=0.9345
    ]
    r = retrievers[-1]
    preds = r.run(data)
    write_preds(preds, path_predict)
    if data_split != SplitEnum.test:
        Scorer().run(data.path_gold, Path(path_predict))
        ResultAnalyzer().run(data, preds)


if __name__ == "__main__":
    Fire(main)
