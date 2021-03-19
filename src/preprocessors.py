from pathlib import Path
from typing import List, Optional, Union, Dict, Set

import pandas as pd
import spacy
from pydantic import BaseModel
from spacy.lang.en import English
from spacy.tokens import Token, Span, Doc
from torchvision.datasets.utils import download_and_extract_archive

from dataset import Statement, TxtAndKeywords, QuestionAnswer
from rankers import deduplicate


class TextProcessor(BaseModel):
    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        return x.raw_txt


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


class OnlyGoldWordsProcessor(TextProcessor):
    """
    What if we can filter the input to only keywords present in gold explanations? (i.e. this is cheating)
    """

    base_processor: SpacyProcessor = SpacyProcessor(remove_stopwords=True)
    questions: List[QuestionAnswer]
    statements: List[Statement]
    qn_to_explains: Dict[str, str] = {}
    qn_to_answer: Dict[str, str] = {}
    add_query_words: bool = True
    add_all_gold_words: bool = False

    def load(self):
        if not self.qn_to_explains:
            uid_to_text = {s.uid: s.raw_txt for s in self.statements}
            for q in self.questions:
                explains = []
                for e in q.explanation_gold:
                    text = uid_to_text.get(e.uid)
                    if text is not None:
                        explains.append(text)
                    else:
                        print(dict(missing_uid=e.uid))
                if not explains:
                    print(dict(qn_no_explains=q.question_id))
                self.qn_to_explains[q.question.raw_txt] = " ".join(explains)

        if not self.qn_to_answer:
            for q in self.questions:
                self.qn_to_answer[q.question.raw_txt] = q.answers[0].raw_txt

    @staticmethod
    def process_words(words: List[str]) -> List[str]:
        words = [w for w in words if w.isalnum()]
        words = [w.lower() for w in words]
        words = sorted(set(words))
        return words

    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        self.load()

        if isinstance(x, Statement):
            return self.base_processor.run(x)
        elif x.raw_txt in self.qn_to_answer.values():
            return self.base_processor.run(x)
        else:
            explains: str = self.qn_to_explains[x.raw_txt]
            gold = TxtAndKeywords(raw_txt=explains)
            text = " ".join([x.raw_txt, self.qn_to_answer[x.raw_txt]])
            x = TxtAndKeywords(raw_txt=text)
            words = self.process_words(self.base_processor.run(x).split())
            words_gold = self.process_words(self.base_processor.run(gold).split())

            if self.add_query_words and self.add_all_gold_words:
                words = words + words_gold
            elif self.add_query_words:
                words = sorted(set(words_gold).intersection(words))
            elif self.add_all_gold_words:
                words = sorted(set(words_gold).difference(words))
            else:
                raise ValueError

            words = deduplicate(words)
            return " ".join(words)


class SentenceSplitProcessor(TextProcessor):
    max_sentences: int = 3
    base_processor: SpacyProcessor = SpacyProcessor(remove_stopwords=True)
    nlp: Optional[English]

    class Config:
        arbitrary_types_allowed = True

    def load(self):
        if self.nlp is None:
            nlp: English = spacy.blank("en")
            nlp.add_pipe(nlp.create_pipe("sentencizer"))
            self.nlp = nlp

    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        self.load()

        if isinstance(x, Statement):
            text = x.raw_txt
        else:
            span: Span
            doc: Doc = self.nlp(x.raw_txt)
            sents = [span.text.strip() for span in doc.sents]
            sents = [s for s in sents if s]
            sents = sents[-self.max_sentences :]
            text = " ".join(sents)

        return self.base_processor.run(TxtAndKeywords(raw_txt=text))


class KeywordProcessor(TextProcessor):
    def run(self, x: Union[TxtAndKeywords, Statement]) -> str:
        return " ".join(x.keywords)


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
