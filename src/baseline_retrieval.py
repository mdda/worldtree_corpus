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
from torch import nn, Tensor
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
from rankers import Ranker, StageRanker, WordEmbedRanker, deduplicate
from vectorizers import BM25Vectorizer, SpacyVectorizer

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

    def run_map_loss(self, data: Data, preds: List[Prediction]):
        qns, preds = self.filter_qns(data, preds)
        uids = deduplicate([s.uid_base for s in data.statements])
        print(len(uids))
        uid_to_i = {u: i for i, u in enumerate(uids)}
        array_gold = np.zeros(shape=(len(preds), len(uids)))
        array_pred = np.copy(array_gold)

        for i, q in enumerate(qns):
            for e in q.explanation_gold:
                j = uid_to_i.get(e.uid)
                if j is not None:
                    array_gold[i, j] = 1
                else:
                    print(dict(uid_not_in_statements=e.uid))

        for i, p in enumerate(preds):
            for j, u in enumerate(p.uids):
                score = 1 / (j + 1)  # First item has highest score
                assert 0.0 <= score <= 1.0
                array_pred[i, uid_to_i[u]] = score

        loss: Tensor = self.loss_fn(
            torch.from_numpy(array_pred).float(), torch.from_numpy(array_gold).float()
        )
        print(dict(map_loss=loss.item()))

    def run(self, data: Data, preds: List[Prediction]):
        self.run_map_loss(data, preds)
        for threshold in self.thresholds:
            recall = self.run_threshold(data, preds, threshold)
            print(dict(threshold=threshold, recall=recall))


class WordEmbedRetriever(Retriever):
    preproc: TextProcessor = SpacyProcessor(remove_stopwords=True)
    vectorizer: TfidfVectorizer = SpacyVectorizer()
    ranker: Ranker = WordEmbedRanker()


def main(data_split=SplitEnum.dev):
    data = Data(data_split=data_split)
    data.load()
    data.analyze()

    # retriever = Retriever()  # Dev MAP=0.3965, recall@512=0.7911
    # retriever = Retriever(preproc=SpacyProcessor())  # Dev MAP=0.4587, recall@512=0.8849
    # retriever = Retriever(
    #     preproc=SpacyProcessor(remove_stopwords=True)
    # )  # Dev MAP=0.4615, recall@512=0.8780
    # retriever = Retriever(preproc=KeywordProcessor())  # Dev MAP=0.4529, recall@512=0.8755
    # retriever = Retriever(
    #     preproc=KeywordProcessor(), ranker=StageRanker()
    # )  # Dev MAP=0.4575, recall@512=0.9095

    # # Maybe this dense vectorizer can make useful features for deep learning methods
    # from vectorizers import TruncatedSVDVectorizer
    # retriever = Retriever(
    #     vectorizer=TruncatedSVDVectorizer(BM25Vectorizer(), n_components=768),
    #     preproc=KeywordProcessor(),
    # )  # Dev MAP:  0.3741, recall@512= 0.8772

    if False:
        retriever = Retriever(
            preproc=KeywordProcessor(), ranker=IterativeRanker()
        )  # Dev MAP=0.4704, recall@512=0.8910
    if True:
        retriever = Retriever(
            preproc=KeywordProcessor(),
            ranker=StageRanker(num_per_stage=[16, 32, 64, 128], scale=1.5),
        )  # Dev MAP=0.4586, recall@512=0.9242

    # retriever = WordEmbedRetriever()  # Dev MAP=0.01436, recall@512=0.4786

    preds = retriever.run(data)
    Scorer().run(data.path_gold, preds)
    ResultAnalyzer().run(data, preds)


if __name__ == "__main__":
    Fire(main)
