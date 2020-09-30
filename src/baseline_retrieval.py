import json
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from fire import Fire
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from tqdm import tqdm

from dataset import Statement, QuestionAnswer, load_qanda, load_statements
from extra_data import SplitEnum, analyze_lengths
from losses import APLoss
from preprocessors import (
    TextProcessor,
    SpacyProcessor,
    SentenceSplitProcessor,
    KeywordProcessor,
    OnlyGoldWordsProcessor,
)
from rankers import Ranker, StageRanker, IterativeRanker, deduplicate
from vectorizers import BM25Vectorizer, TruncatedSVDVectorizer

sys.path.append("../tg2020task")
import evaluate


class Data(BaseModel):
    root: str = "../data"
    root_gold: str = "../tg2020task"
    data_split: SplitEnum = SplitEnum.dev
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
        num_explains = [len(q.explanation_gold) for q in self.questions]
        info = dict(
            statements=len(self.statements),
            questions=len(self.questions),
            num_explains=analyze_lengths(num_explains),
        )
        if self.data_split != SplitEnum.test:
            len_explains: List[int] = []
            for q in self.questions:
                for e in q.explanation_gold:
                    statements = self.uid_to_statements.get(e.uid)
                    if statements is None:
                        print(dict(missing_uid=e.uid))
                    else:
                        state = self.uid_to_statements[e.uid][0]
                        len_explains.append(len(state.raw_txt))
            len_qns = [len(q.question.raw_txt) for q in self.questions]
            info.update(
                len_explains=analyze_lengths(len_explains),
                len_qns=analyze_lengths(len_qns),
            )
        print(info)


class Prediction(BaseModel):
    qid: str
    uids: List[str]


class Retriever(BaseModel):
    preproc: TextProcessor = TextProcessor()
    vectorizer: TfidfVectorizer = BM25Vectorizer()
    ranker: Ranker = Ranker()
    limit: int = -1

    class Config:
        arbitrary_types_allowed = True

    def make_pred(self, i_query: int, rank: List[int], data: Data) -> Prediction:
        uids = [data.statements[i].uid_base for i in rank]
        uids = deduplicate(uids, limit=self.limit)
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
        for i in tqdm(range(len(ranking)), desc="Retriever"):
            preds.append(self.make_pred(i, list(ranking[i]), data))
        return preds


class PredictManager(BaseModel):
    file_pattern: str
    fold_marker: str = "FOLD"
    sep_line: str = "\n"
    sep_field: str = "\t"

    def make_path(self, data_split: str) -> Path:
        assert self.file_pattern.count(self.fold_marker) == 1
        path = Path(self.file_pattern.replace(self.fold_marker, data_split))
        path.parent.mkdir(exist_ok=True)
        return path

    def write(self, preds: List[Prediction], data_split: str):
        lines = []
        for p in preds:
            for u in p.uids:
                lines.append(self.sep_field.join([p.qid, u]))

        path = self.make_path(data_split)
        with open(path, "w") as f:
            f.write(self.sep_line.join(lines))
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(preds, f)

    def read(self, data_split: str) -> List[Prediction]:
        start = time.time()
        qid_to_uids = {}
        path = self.make_path(data_split)
        with open(path) as f:
            for line in f:
                line = line.strip()
                qid, uid = line.split(self.sep_field)
                qid_to_uids.setdefault(qid, []).append(uid)
        preds = [Prediction(qid=qid, uids=uids) for qid, uids in qid_to_uids.items()]
        duration = time.time() - start
        print(dict(read=path, duration=round(duration, 3)))
        return preds

    def read_pickle(self, data_split: str) -> List[Prediction]:
        # About 50x faster than reading .txt, with 7x smaller file
        start = time.time()
        path = self.make_path(data_split).with_suffix(".pkl")
        with open(path, "rb") as f:
            preds = pickle.load(f)

        duration = time.time() - start
        print(dict(read=path, duration=round(duration, 3)))
        return preds


class Scorer(BaseModel):
    @staticmethod
    def run(path_gold: Path, path_predict: Path) -> Dict[str, float]:
        gold = evaluate.load_gold(str(path_gold))  # noqa
        pred = evaluate.load_pred(str(path_predict))  # noqa
        qid2score = {}

        def _callback(qid, score):
            qid2score[qid] = score

        score_fn = evaluate.mean_average_precision_score  # noqa
        mean_ap = score_fn(gold, pred, callback=_callback)
        print(dict(mean_ap=mean_ap))
        with open("/tmp/per_q.json", "wt") as f:
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
        if top_n is None:
            top_n = len(preds[0].uids)
        else:
            top_n = min(len(preds[0].uids), top_n)

        qns, preds = self.filter_qns(data, preds)
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
    manager = PredictManager(file_pattern=output_pattern)

    retrievers = [
        Retriever(),  # Dev MAP=0.3965, recall@512=0.7911
        Retriever(preproc=SpacyProcessor()),  # Dev MAP=0.4587, recall@512=0.8849
        Retriever(
            preproc=SpacyProcessor(remove_stopwords=True)
        ),  # Dev MAP=0.4615, recall@512=0.8780
        Retriever(  # This shows that only the last 4 sentences are necessary...
            preproc=SentenceSplitProcessor(max_sentences=4),
        ),  # Dev MAP=0.4615, recall@512=0.8795
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
        Retriever(  # This shows that keyword generation models have potential
            preproc=OnlyGoldWordsProcessor(
                questions=data.questions,
                statements=data.statements,
                add_all_gold_words=True,
            ),
        ),  # Dev MAP=0.8118, recall@512=1.000
        Retriever(
            preproc=SpacyProcessor(remove_stopwords=True),
            ranker=StageRanker(num_per_stage=[1, 2, 4, 8, 16], scale=1.25),
            vectorizer=BM25Vectorizer(binary=True, use_idf=True, k1=2.0, b=0.5),
        ),  # Dev MAP=0.4861, recall@512=0.9345
    ]
    r = retrievers[-1]
    preds = r.run(data)
    manager.write(preds, data_split)
    if data_split != SplitEnum.test:
        Scorer().run(data.path_gold, manager.make_path(data_split))
        ResultAnalyzer().run(data, preds)


if __name__ == "__main__":
    Fire(main)
