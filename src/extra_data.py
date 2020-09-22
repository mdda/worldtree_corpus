import bz2
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple, Set
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url, download_and_extract_archive
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding
from wikiextractor import WikiExtractor

KEY_LABEL = "labels"


class PickleSaver(BaseModel):
    path: Path

    def dump(self, obj: Any):
        with open(str(self.path), "wb") as f:
            pickle.dump(obj, f)

    def load(self) -> Any:
        with open(str(self.path), "rb") as f:
            return pickle.load(f)


class SplitEnum(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"
    all = "all"

    @classmethod
    def select(cls, train: list, dev: list, test: list, data_split: str) -> list:
        assert isinstance(data_split, cls)
        mapping = {
            cls.train: train,
            cls.dev: dev,
            cls.test: test,
            cls.all: train + dev + test,
        }
        return mapping[data_split]


def extract_bz2(path_in: Path, path_out: Path = None) -> Path:
    if path_out is None:
        path_out = path_in.with_suffix("")  # Eg test.xml.bz2 -> test.xml
    if not path_out.exists():
        with bz2.open(path_in, "rb") as f:
            content = f.read()
        with open(str(path_out), "wb") as f:
            f.write(content)
    return path_out


def run_script_with_kwargs(fn: Callable, args: List[str], kwargs: Dict[str, str]):
    args = [""] + args
    for k, v in kwargs.items():
        args.append(f"--{k}")
        args.append(v)

    with patch.object(sys, "argv", args):
        fn()


def glob_single(path: Path, pattern: str) -> Path:
    candidates = sorted(path.glob(pattern))
    return candidates[0]


def read_lines(path: Path, limit: int) -> List[str]:
    lines_all = []
    with open(str(path)) as f:
        for line in tqdm(f, total=limit):
            lines_all.append(line)
            if len(lines_all) >= limit:
                break
    return lines_all


def analyze_lengths(lengths: List[int]) -> Dict[str, float]:
    return dict(mean=np.mean(lengths), min=np.min(lengths), max=np.max(lengths), std=np.std(lengths))


def train_dev_test_split(
    items: list, train_fraction=0.8, do_shuffle=True, random_seed=42
) -> Tuple[list, list, list]:
    # Assumes dev_fraction == test_fraction eg 80-10-10 split
    assert 0.0 <= train_fraction <= 1.0
    train, dev_test = train_test_split(
        items, train_size=train_fraction, shuffle=do_shuffle, random_state=random_seed
    )
    dev, test = train_test_split(
        dev_test, train_size=0.5, shuffle=do_shuffle, random_state=random_seed
    )
    return train, dev, test


def format_query_and_evidence(query: str, evidence: str) -> str:
    return f"Query: {query} Evidence: {evidence}"


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


class MSMarcoExample(BaseModel):
    qid: str
    pids: List[str]
    labels: List[int]


class MSMarcoPassageRankData(BaseModel):
    """
    https://github.com/microsoft/MSMARCO-Passage-Ranking

    Given a query q and a the 1000 most relevant passages P = p1, p2, p3,... p1000,
    as retrieved by BM25 a successful system is expected to rerank the most relevant passage
    as high as possible. For this task not all 1000 relevant items have a human labeled
    relevant passage. Evaluation will be done using MRR
    """

    url: str = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
    url_triples: str = "https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.tsv.gz"
    root: str = "/tmp/ms_marco_passage_rank"
    data_split: SplitEnum = SplitEnum.train
    examples: Optional[List[MSMarcoExample]]
    pid_to_text: Optional[Dict[str, str]]
    qid_to_text: Optional[Dict[str, str]]
    limit: int = int(1e7)
    sep: str = "\t"

    def read_triples(self, path) -> List[MSMarcoExample]:
        qid_to_pids: Dict[str, List[str]] = {}
        qid_to_labels: Dict[str, List[int]] = {}

        name = f"{self.limit}_{path.name}"
        saver = PickleSaver(path=path.with_name(name).with_suffix(".pkl"))
        if saver.path.exists():
            return saver.load()

        for line in tqdm(read_lines(path, self.limit)):
            line = line.strip()
            assert line
            qid, pid_pos, pid_neg = line.split()
            for pid, label in [(pid_pos, 1), (pid_neg, 0)]:
                qid_to_pids.setdefault(qid, []).append(pid)
                qid_to_labels.setdefault(qid, []).append(label)

        assert qid_to_pids.keys() == qid_to_labels.keys()
        examples = []
        for qid, pids in tqdm(list(qid_to_pids.items())):
            examples.append(
                MSMarcoExample(qid=qid, pids=pids, labels=qid_to_labels[qid])
            )
        assert examples
        saver.dump(examples)
        return examples

    def read_id_to_text(self, path: Path, ids: Set[str]) -> Dict[str, str]:
        hash_id = hash_text(str(sorted(ids)))
        saver = PickleSaver(path=Path(self.root) / f"id_to_text_{hash_id}.pkl")
        if saver.path.exists():
            return saver.load()

        id_to_text: Dict[str, str] = {}
        with open(path) as f:
            lines_all = f.readlines()
        for line in tqdm(lines_all):
            line = line.strip()
            assert line
            i, text = line.split(self.sep)
            if ids is None or i in ids:
                id_to_text[i] = text
        assert ids == set(id_to_text.keys())
        saver.dump(id_to_text)
        return id_to_text

    def load(self):
        root = Path(self.root)
        if not root.exists():
            download_and_extract_archive(self.url, root, root)
            download_url(self.url_triples, root)
            subprocess.run(["gunzip", str(root / Path(self.url_triples).name)])
            # download_and_extract_archive(self.url_triples, root, root)  # OOM
        examples = self.read_triples(root / "qidpidtriples.train.full.tsv")
        train, dev, test = train_dev_test_split(examples)
        self.examples = SplitEnum.select(train, dev, test, self.data_split)
        self.qid_to_text = self.read_id_to_text(
            root / "queries.train.tsv", ids=set([e.qid for e in self.examples])
        )
        self.pid_to_text = self.read_id_to_text(
            root / "collection.tsv", ids=set([i for e in self.examples for i in e.pids])
        )

    def analyze(self):
        assert self.examples
        print(type(self).__name__)
        info_examples = dict(
            qids=len(self.examples),
            pids_per_qid=analyze_lengths([len(e.pids) for e in self.examples]),
            pids=len(set([p for e in self.examples for p in e.pids])),
            labels=Counter([lab for e in self.examples for lab in e.labels]),
            words=analyze_lengths(
                [
                    len(self.pid_to_text[p].split())
                    for e in self.examples
                    for p in e.pids
                ]
            ),
        )
        print(info_examples)


class MSMarcoHierarchicalSentenceDataset(Dataset):
    def __init__(
        self,
        data_split: SplitEnum,
        tokenizer: PreTrainedTokenizer,
        max_seq_per_example=128,
        max_seq_len=128,
    ):
        self.max_seq_per_example = max_seq_per_example
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data = MSMarcoPassageRankData(data_split=data_split)
        self.data.load()

    def __getitem__(self, i: int) -> BatchEncoding:
        e = self.data.examples[i]
        query = self.data.qid_to_text[e.qid]
        texts = []

        assert len(e.pids) == len(e.labels)
        indices = random.choices(range(len(e.pids)), k=self.max_seq_per_example)
        pids = [e.pids[i] for i in indices]
        labels = [e.labels[i] for i in indices]

        for p in pids:
            evidence = self.data.pid_to_text[p]
            texts.append(format_query_and_evidence(query, evidence))
        x: BatchEncoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        x.data[KEY_LABEL] = Tensor(labels).long()
        return x

    def __len__(self) -> int:
        return len(self.data.examples)

    def get_loader(self, bs=1) -> DataLoader:
        return DataLoader(
            self,
            batch_size=bs,
            shuffle=(self.data.data_split == SplitEnum.train),
            num_workers=os.cpu_count(),
        )

    def analyze(self):
        loader = self.get_loader()
        batches = [next(iter(loader)) for _ in range(3)]
        print(type(self).__name__, "DataLoader")
        for x in batches:
            print(dict(x={k: v.shape for k, v in x.items()}))


class SimpleWikiExample(BaseModel):
    id: int
    url: str
    title: str
    text: str


class SimpleWikiData(BaseModel):
    url_articles: str = "https://dumps.wikimedia.org/simplewiki/20200901/simplewiki-20200901-pages-articles-multistream.xml.bz2"
    root: str = "/tmp/simple_wikipedia"
    examples: Optional[List[SimpleWikiExample]]

    def download(self, url: str) -> Path:
        filename = Path(url).name
        path = Path(self.root) / filename
        if not path.exists():
            download_url(url, self.root, filename)
        assert path.exists()

        if path.suffix == ".bz2":
            path = extract_bz2(path)
        return path

    def load(self):
        path_articles = self.download(self.url_articles)
        dir_extract = Path(self.root) / "articles"
        if not dir_extract.exists():
            run_script_with_kwargs(
                fn=WikiExtractor.main,
                args=["--json", str(path_articles)],
                kwargs=dict(output=str(dir_extract)),
            )

        if not self.examples:
            self.examples = []
            for path in tqdm(sorted(dir_extract.glob("*/wiki_*"))):
                with open(str(path)) as f:
                    for line in f:
                        self.examples.append(SimpleWikiExample(**json.loads(line)))

        assert len(set([e.title for e in self.examples])) == len(self.examples)
        for e in self.examples:
            assert e.text.startswith(e.title), dict(title=e.title, text=e.text)
            e.text = e.text[len(e.title) :]

    def analyze(self):
        self.load()
        print(type(self).__name__)
        print(dict(length=len(self.examples)))

        random.seed(42)
        samples = random.sample(self.examples, k=10)
        for s in samples:
            record = s.dict()
            print(json.dumps(record, indent=2))


class AristoMiniCorpus(BaseModel):
    """
    The Aristo Mini corpus contains 1,197,377 (very loosely) science-relevant
    sentences drawn from public data. It provides simple science-relevant text
    that may be useful to help answer elementary science questions.
    """

    url: str = "https://ai2-datasets.s3-us-west-2.amazonaws.com/aristo-mini/Aristo-Mini-Corpus-Dec2016.zip"
    root: str = "/tmp/aristo_mini_corpus"
    data_split: SplitEnum = SplitEnum.all
    texts: Optional[List[str]]

    def load(self):
        if not Path(self.root).exists():
            download_and_extract_archive(self.url, self.root, self.root)

        path = glob_single(Path(self.root), "**/Aristo-Mini-Corpus-Dec2016.txt")
        if not self.texts:
            with open(path) as f:
                texts = sorted(set([line.strip() for line in f if line.strip()]))
            train, dev, test = train_dev_test_split(texts)
            self.texts = SplitEnum.select(train, dev, test, self.data_split)

    def analyze(self):
        self.load()
        print(type(self).__name__)

        random.seed(42)
        samples = random.sample(self.texts, k=10)
        info = dict(
            texts=len(self.texts),
            lengths=analyze_lengths([len(x.split()) for x in self.texts]),
            samples=samples,
        )
        print(info)


class MultiNLIExample(BaseModel):
    annotator_labels: List[str]
    genre: str
    gold_label: str
    pairID: str
    promptID: str
    sentence1: str
    sentence1_binary_parse: str
    sentence1_parse: str
    sentence2: str
    sentence2_binary_parse: str
    sentence2_parse: str


class MultiNLIData(BaseModel):
    root: str = "/tmp/multi_nli"
    url: str = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    data_split: SplitEnum = SplitEnum.train
    examples: Optional[List[MultiNLIExample]]

    def load(self):
        if not Path(self.root).exists():
            download_and_extract_archive(self.url, self.root, self.root)

        suffix = {
            SplitEnum.train: "train",
            SplitEnum.dev: "dev_matched",
            SplitEnum.test: "dev_match",
        }[self.data_split]
        name = f"multinli_1.0_{suffix}.jsonl"
        path = glob_single(Path(self.root), f"**/{name}")

        if not self.examples:
            with open(path) as f:
                self.examples = [MultiNLIExample(**json.loads(line)) for line in f]

    def analyze(self):
        self.load()
        print(type(self).__name__)
        samples = random.sample(self.examples, k=10)
        info = dict(length=len(self.examples), samples=[s.dict() for s in samples])
        print(json.dumps(info, indent=2))


class SciTailExample(BaseModel):
    answer: str
    gold_label: str
    question: str
    sentence1: str
    sentence2: str
    sentence2_structure: str

    @property
    def statement(self) -> str:
        return self.sentence2

    @property
    def evidence(self) -> str:
        return self.sentence1

    @property
    def has_entail(self) -> bool:
        return self.gold_label == "entails"


class SciTailData(BaseModel):
    """
    The SciTail dataset is an entailment dataset created from multiple-choice
    science exams and web sentences. Each question and the correct answer
    choice are converted into an assertive statement to form the hypothesis.
    """

    url: str = "http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip"
    root: str = "/tmp/scitail"
    data_split: SplitEnum = SplitEnum.train
    examples: Optional[List[SciTailExample]]

    def load(self):
        if not Path(self.root).exists():
            download_and_extract_archive(self.url, self.root, self.root)

        if not self.examples:
            name = f"scitail_1.0_structure_{self.data_split}.jsonl"
            path = glob_single(Path(self.root), f"**/predictor_format/{name}")
            with open(path) as f:
                self.examples = [SciTailExample(**json.loads(line)) for line in f]

    def analyze(self):
        self.load()
        print(dict(labels=Counter([e.gold_label for e in self.examples])))
        random.seed(42)
        for e in random.sample(self.examples, k=3):
            print(
                json.dumps(
                    dict(
                        question=e.question,
                        answer=e.answer,
                        evidence=e.evidence,
                        entail=e.has_entail,
                        statement=e.statement,
                    ),
                    indent=2,
                )
            )
        print()


class TextGraphsQuestion(BaseModel):
    QuestionID: str
    originalQuestionID: str
    totalPossiblePoint: int
    AnswerKey: str
    isMultipleChoiceQuestion: int
    includesDiagram: int
    examName: str
    schoolGrade: int
    subject: str
    category: str
    year: str
    topic: str
    explanation: str
    question: str
    flags: str
    arcset: str


class TextGraphsExplanation(BaseModel):
    category: str
    uid: str
    text: str
    raw: Dict[str, str]


class TextGraphsData(BaseModel):
    root: str = "../tg2020task"
    data_split: SplitEnum = SplitEnum.train
    questions: Optional[List[TextGraphsQuestion]]
    explanations: Optional[List[TextGraphsExplanation]]

    @staticmethod
    def read_explanations(path: Path) -> List[TextGraphsExplanation]:
        # Reference: https://github.com/cognitiveailab/tg2020task/blob/master/baseline_tfidf.py
        cols_text = []
        col_uid = None
        df = pd.read_csv(path, sep="\t", dtype=str)

        for name in df.columns:
            if name.startswith("[SKIP]"):
                if "UID" in name and not col_uid:
                    col_uid = name
            else:
                cols_text.append(name)

        if not col_uid or len(df) == 0:
            warnings.warn("Possibly misformatted file: " + str(path))
            return []

        explanations = []
        for r in df.to_dict(orient="records"):
            raw = {h: r[h] for h in cols_text if pd.notna(r[h])}
            explanations.append(
                TextGraphsExplanation(
                    category=path.stem,
                    uid=r[col_uid],
                    text=" ".join(list(raw.values())),
                    raw=raw,
                )
            )
        return explanations

    def load(self):
        path_questions = Path(self.root) / f"questions.{self.data_split}.tsv"
        df = pd.read_csv(path_questions, sep="\t")
        self.questions = [TextGraphsQuestion(**r) for r in df.to_dict(orient="records")]

        self.explanations = []
        for p in (Path(self.root) / "tables").iterdir():
            self.explanations.extend(self.read_explanations(p))

    def analyze(self):
        self.load()
        info = dict(
            questions=dict(
                num=len(self.questions),
                len=analyze_lengths([len(x.question.split()) for x in self.questions]),
            ),
            explanations=dict(
                num=len(self.explanations),
                len=analyze_lengths([len(x.text.split()) for x in self.explanations]),
            ),
        )
        print(info)


def main():
    # for s in [SplitEnum.train, SplitEnum.dev, SplitEnum.test]:
    #     data = SciTailData(data_split=s)
    #     data.load()
    #     data.analyze()
    #
    # corpus = AristoMiniCorpus()
    # corpus.analyze()
    #
    # data = MultiNLI()
    # data.analyze()
    #
    data = SimpleWikiData()
    data.analyze()

    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # dataset = MSMarcoHierarchicalSentenceDataset(SplitEnum.train, tokenizer)
    # dataset.analyze()
    # dataset.data.analyze()

    # data = TextGraphsData()
    # data.analyze()


if __name__ == "__main__":
    main()
