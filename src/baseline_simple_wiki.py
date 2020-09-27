from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import spacy
from pydantic import BaseModel
from pytorch_lightning.loggers import CometLogger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from spacy.lang.en import English
from tqdm import tqdm

from baseline_rerank import (
    System,
    RerankDataset,
    Config,
    get_logger,
    run_train,
    run_eval,
)
from baseline_retrieval import Data, PredictManager, Retriever, ResultAnalyzer
from baseline_retrieval import Prediction  # noqa
from dataset import Statement, QuestionAnswer, ExplanationUsed, TxtAndKeywords
from extra_data import (
    SimpleWikiData,
    PickleSaver,
    hash_text,
    train_dev_test_split,
    SplitEnum,
)
from rankers import LimitRanker

"""
Simple Wikipedia Pre-Training
For each article in SW
    Check if have overlap with TextGraphs
    Split the text into sentences (strip whitespace?)
    Check if too little sentences
    Set the query as first sentence in the paragraph
    Goal: Retrieve gold sentences in the presence of distractions
    The distractions will be related sentences (via IR)
    Architecture will be HierarchicalRankNet
    Optimize via MAP loss function.
"""


class Sentences(BaseModel):
    texts: List[str]

    @property
    def qn(self) -> str:
        return self.texts[0]

    @property
    def docs(self) -> List[str]:
        return self.texts[1:]

    @property
    def is_valid(self) -> bool:
        return len(self.texts) >= 2


class SentenceSplitter(BaseModel):
    spacy_model: str = "en_core_web_sm"
    cache_dir: str = "/tmp/sentence_splitter"

    @staticmethod
    def split_texts(texts: List[str]) -> List[Sentences]:
        # # https://github.com/explosion/spaCy/issues/3842
        nlp: English = spacy.blank("en")
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        sents = []
        for doc in tqdm(nlp.pipe(texts), total=len(texts), desc="split_texts"):
            texts = [s.text.strip() for s in doc.sents if s.text.strip()]
            sents.append(Sentences(texts=texts))
        return sents

    def run(self, texts: List[str]) -> List[Sentences]:
        path = Path(self.cache_dir) / f"{hash_text(' '.join(texts))}.pkl"
        path.parent.mkdir(exist_ok=True)
        saver = PickleSaver(path=path)

        if saver.path.exists():
            print(dict(path_cache=saver.path))
            return saver.load()
        else:
            sents = self.split_texts(texts)
            saver.dump(sents)
            return sents


class WikiData(Data):
    data_wiki: SimpleWikiData = SimpleWikiData()
    data_base: Data = Data()
    vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words="english")
    max_qns: int = 8000
    max_docs_per_qn: int = 32
    max_words_per_doc: int = 32
    splitter: SentenceSplitter = SentenceSplitter()

    class Config:
        arbitrary_types_allowed = True

    def truncate_sents(self, sents: List[Sentences]) -> List[Sentences]:
        sents = deepcopy(sents)
        for s in sents:
            texts = [" ".join(t.split()[: self.max_words_per_doc]) for t in s.texts]
            assert len(texts) == len(s.texts)
            s.texts = texts
        return sents

    def sents_to_statements(self, sents) -> List[Statement]:
        docs = [d for s in sents for d in s.docs[: self.max_docs_per_qn]]
        docs = sorted(set(docs))
        statements = []
        for d in tqdm(docs, desc="texts_to_statements"):
            uid = hash_text(d)
            s = Statement(
                uid_base=uid,
                uid=uid,
                table="",
                hdr_arr=[],
                txt_arr=[],
                keyword_arr=[],
                raw_txt=d,
                keywords=set(),
            )
            statements.append(s)
        return statements

    def sents_to_qns(self, sents: List[Sentences]) -> List[QuestionAnswer]:
        assert self.uid_to_statements
        assert all([len(x) == 1 for x in self.uid_to_statements.values()])
        text_to_uid = {x[0].raw_txt: uid for uid, x in self.uid_to_statements.items()}

        qns: List[QuestionAnswer] = []
        s: Sentences
        for s in tqdm(sents, desc="sents_to_qns"):
            docs = s.docs[: self.max_docs_per_qn]
            explains = [ExplanationUsed(uid=text_to_uid[d], reason="") for d in docs]
            q = QuestionAnswer(
                question_id=hash_text(s.qn),
                question=TxtAndKeywords(raw_txt=s.qn),
                answers=[TxtAndKeywords(raw_txt="")],
                explanation_gold=explains,
            )
            qns.append(q)

        return qns

    @staticmethod
    def filter_sents(sents: List[Sentences]) -> List[Sentences]:
        filtered = [s for s in sents if s.is_valid]
        print(dict(sents_too_short=len(sents) - len(filtered)))
        return filtered

    def load(self):
        self.data_wiki.load()
        self.data_base.load()

        texts = [e.text for e in self.data_wiki.examples]
        query = " ".join([s.raw_txt for s in self.data_base.statements])
        self.vectorizer.fit([query])
        distances = cosine_distances(
            self.vectorizer.transform([query]), self.vectorizer.transform(texts)
        )
        distances = np.squeeze(distances)
        ranking = np.argsort(distances)
        examples = [self.data_wiki.examples[i] for i in ranking[: self.max_qns]]
        sents = self.splitter.run([e.text for e in examples])
        sents = self.filter_sents(sents)
        sents = self.truncate_sents(sents)

        self.statements = self.sents_to_statements(sents)
        self.uid_to_statements = {s.uid: [s] for s in self.statements}

        train, dev, test = train_dev_test_split(sents)
        sents = SplitEnum.select(train, dev, test, self.data_split)
        self.questions = self.sents_to_qns(sents)
        assert all([len(q.explanation_gold) > 0 for q in self.questions])


class WikiSystem(System):
    @staticmethod
    def make_initial_preds(
        data: WikiData, output_pattern: str, top_n=512,
    ):
        manager = PredictManager(file_pattern=output_pattern)
        if not manager.make_path(data.data_split).exists():
            r = Retriever(ranker=LimitRanker(top_n=top_n), limit=top_n)
            preds = r.run(data)
            manager.write(preds, data.data_split)
            if data.data_split != SplitEnum.test:
                ResultAnalyzer().run(data, preds)

    def make_dataset(self, data_split: SplitEnum) -> RerankDataset:
        path_cache = Path(f"/tmp/wiki_system/{data_split}.pkl")
        if not path_cache.parent.exists():
            path_cache.parent.mkdir(exist_ok=True)
        saver = PickleSaver(path=path_cache)
        if saver.path.exists():
            data = saver.load()
            print(dict(path_cache=path_cache))
        else:
            data = WikiData(data_split=data_split)
            data.load()
            saver.dump(data)

        data.analyze()
        self.make_initial_preds(data, output_pattern=self.config.input_pattern)
        return RerankDataset(data, self.config, is_test=(data_split == SplitEnum.test))


def run_train_base(system: WikiSystem, logger: CometLogger):
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=system.config.num_epochs,
        overfit_batches=system.config.overfit_pct,
        gpus=1,
    )  # precision=16 is not faster on P100
    trainer.fit(system)


def main_base(save_dir: str, path_dotenv: str):
    config = Config(
        input_pattern="/tmp/wiki_system/predict.FOLD.baseline-retrieval.txt",
        output_pattern="/tmp/wiki_system/predict.FOLD.baseline-rerank.txt",
    )
    if not Path(save_dir).exists():
        system = WikiSystem(config.dict())
        logger = get_logger(save_dir, path_dotenv)
        run_train_base(system, logger)


def main(
    base_dir="/tmp/wiki_system/comet_logger",
    save_dir="comet_logger",
    path_dotenv="../excluded/.env",
):
    main_base(base_dir, path_dotenv)
    path_load = list(Path(base_dir).glob("**/*.ckpt"))[0]
    print(dict(path_load=path_load))
    system_base = WikiSystem.load_from_checkpoint(str(path_load))

    if not Path(save_dir).exists():
        logger = get_logger(save_dir, path_dotenv)
        run_train(logger, state=system_base.state_dict())

    run_eval(save_dir, data_split=SplitEnum.dev)
    run_eval(save_dir, data_split=SplitEnum.test)


if __name__ == "__main__":
    main()
