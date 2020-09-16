import random
from typing import List, Dict, Union

import pandas as pd
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from baseline_retrieval import (
    Data,
    Retriever,
    ResultAnalyzer,
    SpacyProcessor,
    TextProcessor,
    KeywordProcessor,
)
from extra_data import SplitEnum
from rankers import StageRanker
from vectorizers import BM25Vectorizer

BasicValue = Union[str, float, int, bool]


class GridOptions(BaseModel):
    preproc_name: List[str] = ["keyword", "spacy"]
    stage_ranker_scale: List[float] = [1.25, 1.5, 1.75]
    stage_ranker_nps_multiplier: List[int] = [1, 2]
    bm25_binary: List[bool] = [True]
    bm25_use_idf: List[bool] = [True]
    bm25_k1: List[float] = [1.5, 2.0, 2.5]
    bm25_b: List[float] = [0.5, 0.75, 1.0]


class Config(BaseModel):
    preproc_name: str
    stage_ranker_scale: float
    stage_ranker_nps_multiplier: int
    bm25_binary: bool
    bm25_use_idf: bool
    bm25_k1: float
    bm25_b: float

    def make_retriever(self) -> Retriever:
        num_per_stage = [1, 2, 4, 8, 16]
        nps = [n * self.stage_ranker_nps_multiplier for n in list(num_per_stage)]
        preproc: TextProcessor = dict(
            keyword=KeywordProcessor(), spacy=SpacyProcessor(remove_stopwords=True)
        )[self.preproc_name]
        ranker = StageRanker(num_per_stage=nps, scale=self.stage_ranker_scale)
        vectorizer = BM25Vectorizer(
            binary=self.bm25_binary,
            use_idf=self.bm25_use_idf,
            k1=self.bm25_k1,
            b=self.bm25_b,
        )
        return Retriever(preproc=preproc, vectorizer=vectorizer, ranker=ranker)


def enumerate_grid(options: Dict[str, List[BasicValue]]) -> List[Dict[str, BasicValue]]:
    key, values = options.popitem()
    if not options:
        return [{key: v} for v in values]

    outputs = []
    record: Dict[str, BasicValue]
    for record in enumerate_grid(options):
        for v in values:
            outputs.append(dict(**record, **{key: v}))
    return outputs


def test_enumerate_grid():
    options = dict(one=[1, 2], two=[3, 4], three=[5, 6, 7])
    records = enumerate_grid(options)
    assert len(set([tuple(r.items()) for r in records])) == 2 * 2 * 3


def main(data_split=SplitEnum.dev):
    data = Data(data_split=data_split)
    data.load()
    data.analyze()

    options = GridOptions()
    configs = [Config(**r) for r in enumerate_grid(options.dict())]
    random.seed(42)
    random.shuffle(configs)
    print(dict(configs=len(configs)))

    analyzer = ResultAnalyzer()
    results = []
    c: Config
    for c in tqdm(configs):
        retriever = c.make_retriever()
        preds = retriever.run(data)
        loss = analyzer.run_map_loss(data, preds)
        results.append(dict(**c.dict(), loss=loss))

    df = pd.DataFrame(results)
    df = df.sort_values(by=["loss"])
    with pd.option_context("display.max_columns", 999):
        print(df.head(10))


if __name__ == "__main__":
    test_enumerate_grid()
    Fire(main)
