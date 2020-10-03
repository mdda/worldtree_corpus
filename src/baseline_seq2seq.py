from pathlib import Path
from typing import List, Optional, Dict

from pydantic import BaseModel

from baseline_retrieval import Data, Retriever, PredictManager, Scorer, ResultAnalyzer
from dataset import QuestionAnswer, TxtAndKeywords
from extra_data import SplitEnum
from preprocessors import TextProcessor, OnlyGoldWordsProcessor, SpacyProcessor


class Seq2SeqData(BaseModel):
    preproc_source: TextProcessor = TextProcessor()
    preproc_target: Optional[TextProcessor]
    data: Data
    root: Path = Path("/tmp/seq_to_seq_data")
    sep: str = "\n"

    def make_source(self, qa: QuestionAnswer):
        q = self.preproc_source.run(qa.question)
        a = self.preproc_source.run(qa.answers[0])
        return f"Question: {q} Answer: {a}"

    def make_target(self, qa: QuestionAnswer) -> str:
        return f"Keywords: {self.preproc_target.run(qa.question)}"

    def load(self):
        self.data.load()

        if self.preproc_target is None:
            self.preproc_target = OnlyGoldWordsProcessor(
                questions=self.data.questions,
                statements=self.data.statements,
                add_query_words=False,
                add_all_gold_words=True,
            )

    def write(self):
        self.load()
        self.root.mkdir(exist_ok=True)

        data_split = {
            SplitEnum.train: "train",
            SplitEnum.dev: "val",
            SplitEnum.test: "test",
        }[self.data.data_split]
        path = self.root / data_split

        with open(path.with_suffix(".source"), "w") as f:
            texts = [self.make_source(qa) for qa in self.data.questions]
            f.write(self.sep.join(texts))
        with open(path.with_suffix(".target"), "w") as f:
            texts = [self.make_target(qa) for qa in self.data.questions]
            f.write(self.sep.join(texts))


class LoadRetriever(Retriever):
    path_load: Path
    queries: Optional[List[str]]
    data: Data
    q_mapping: Optional[Dict[str, str]]

    def load(self):
        if self.q_mapping is None:
            q_orig = [q.question.raw_txt for q in self.data.questions]

            assert self.path_load.exists()
            with open(self.path_load) as f:
                q_new = f.read().split("\n")
                assert len(q_new) == len(q_orig)

            self.q_mapping = {o: n for o, n in zip(q_orig, q_new)}

    def make_query(self, q: QuestionAnswer, data: Data) -> str:
        self.load()
        text = self.q_mapping[q.question.raw_txt]
        prefix = "Keywords: "
        assert text.startswith(prefix)
        text = text[len(prefix) :]

        words = sorted(set(text.split()))
        words = [q.question.raw_txt, q.answers[0].raw_txt] + words
        return self.preproc.run(TxtAndKeywords(raw_txt=" ".join(words)))


def main():
    path = Path("/tmp/seq_to_seq_data/outputs/test_generations.txt")
    if path.exists():
        data_split = SplitEnum.dev
        data = Data(data_split=data_split)
        data.load()
        r = LoadRetriever(
            path_load=path, data=data, preproc=SpacyProcessor(remove_stopwords=True)
        )
        manager = PredictManager(
            file_pattern="../predictions/predict.FOLD.baseline-seq2seq.txt"
        )
        preds = r.run(data)
        manager.write(preds, data_split)
        Scorer().run(data.path_gold, manager.make_path(data_split))
        ResultAnalyzer().run(data, preds)
    else:
        for data_split in [SplitEnum.train, SplitEnum.dev, SplitEnum.test]:
            data = Seq2SeqData(data=Data(data_split=data_split))
            data.write()


if __name__ == "__main__":
    main()
