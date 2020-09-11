import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import nltk
import numpy as np
import pandas as pd
from pydantic import BaseModel
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

sys.path.append("../tg2020task")
np.random.seed(42)
nltk.download("stopwords")

from baseline_tfidf import read_explanations


class TextGraphLemmatizer(BaseModel):
    """
    Works best to transform texts before and also get lemmas during tokenization
    """

    root: str = "/tmp/TextGraphLemmatizer"
    url: str = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/worldtree_textgraphs_2019_010920.zip"
    sep: str = "\t"
    word_to_lemma: Optional[Dict[str, str]]

    def read_csv(
        self, path: Path, header: str = None, names: List[str] = None
    ) -> pd.DataFrame:
        return pd.read_csv(path, header=header, names=names, sep=self.sep)

    def load(self):
        if self.word_to_lemma:
            return

        download_and_extract_archive(self.url, self.root, self.root)
        root_anno = list(Path(self.root).glob("**/annotation"))[0]
        df = self.read_csv(root_anno / "lemmatization-en.txt", names=["lemma", "word"])

        # path_extra = list(Path(self.root).glob("**/LemmatizerAdditions.tsv"))[0]
        # df_extra= self.read_csv(path_extra, names=["lemma", "word", "useless"])
        # df_extra.drop(columns=["useless"], inplace=True)
        # df_extra.dropna(inplace=True)
        # df = pd.concat([df, df_extra])  # Actually concat extra hurts MAP (0.462->0.456)

        def only_alpha(text: str) -> str:
            # Remove punct eg dry-clean -> dryclean so
            # they won't get split by downstream tokenizers
            return "".join([c for c in text if c.isalpha()])

        self.word_to_lemma = {
            word.lower(): only_alpha(lemma.lower()) for word, lemma in df.values
        }

    def lookup(self, word: str) -> str:
        return self.word_to_lemma.get(word, word)

    def transform_text(self, text: str) -> str:
        return " ".join([self.lookup(word) for word in text.split()])

    def transform(self, texts: List[str]) -> list:
        self.load()
        return [self.transform_text(t) for t in texts]


def preprocess_texts(texts: list) -> Tuple[list, list]:
    # NLTK tokenizer on par with spacy and less complicated
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    default_lemmatizer = TextGraphLemmatizer()
    # wordnet_lemmatizer doesn't help
    texts = default_lemmatizer.transform(texts)
    stops = set(nltk.corpus.stopwords.words("english"))

    def lemmatize(token):
        return default_lemmatizer.word_to_lemma.get(token) or token

    def process(
        text: str, _tokenizer: nltk.tokenize.TreebankWordTokenizer,
    ) -> (list, list):
        _tokens = _tokenizer.tokenize(text.lower())
        _lemmas = [
            lemmatize(_tok)
            for _tok in _tokens
            if _tok not in stops and not _tok.isspace()
        ]
        return _tokens, _lemmas

    tokens, lemmas = zip(*[process(text, tokenizer) for text in tqdm(texts)])
    return tokens, lemmas


def exp_skip_dep(
    path_exp: Path, col: str = "[SKIP] DEP", save_temp: bool = True,
) -> str:
    """
  Remove rows that have entries in deprecated column
  according to https://github.com/umanlp/tg2019task/issues/2
  """
    df = pd.read_csv(path_exp, sep="\t")
    if col in df.columns:
        df = df[df[col].isna()]
    path_new = "temp.tsv" if save_temp else Path(path_exp).name
    df.to_csv(path_new, sep="\t", index=False)
    return path_new


def get_df_explanations(path_tables: str):
    """
  Make a dataframe of explanation sentences (~5000)
  """
    explanations = []
    columns = None
    for p in Path(path_tables).iterdir():
        columns = ["uid", "text"]
        p = exp_skip_dep(p)
        # p = save_unique_phrases(Path(p))
        explanations += read_explanations(str(p))
    df = pd.DataFrame(explanations, columns=columns)
    df = df.drop_duplicates("uid").reset_index(drop=True)  # 3 duplicate uids
    tokens, lemmas = preprocess_texts(df.text.tolist())
    df["tokens"], df["lemmas"], df["embedding"] = tokens, lemmas, None
    print("Explanation df shape:", df.shape)
    return df


def extract_explanation(exp_string):
    """
  Convert raw string (eg "uid1|role1 uid2|role2" -> [uid1, uid2], [role1, role2])
  """
    if type(exp_string) != str:
        return [], []
    uids = []
    roles = []
    for uid_and_role in exp_string.split():
        uid, role = uid_and_role.split("|")
        uids.append(uid)
        roles.append(role)
    return uids, roles


def split_question(q_string):
    """
  Split on option parentheses (eg "Question (A) option1 (B) option2" -> [Question, option 1, option2])
  Note that some questions have more or less than 4 options
  """
    return re.compile("\\(.\\)").split(q_string)


def add_q_reformat(df: pd.DataFrame) -> pd.DataFrame:
    q_reformat = []
    questions: List[str] = df["question"].tolist()
    answers: List[str] = df["AnswerKey"].tolist()
    char2idx = {char: idx for idx, char in enumerate(list("ABCDE"))}

    for i in range(len(df)):
        q, *options = split_question(questions[i])
        if answers[i].isdigit():
            idx_option = int(answers[i]) - 1
            assert idx_option >= 0
        else:
            idx_option = char2idx[answers[i]]
        q_reformat.append(" ".join([q.strip(), options[idx_option].strip()]))
    df["q_reformat"] = q_reformat
    return df


def get_questions(path: str, uid2idx: dict = None) -> pd.DataFrame:
    """
  Identify correct answer text and filter out wrong distractors from question string
  Get tokens and lemmas
  Get explanation sentence ids and roles
  """
    # Dropping questions without explanations hurts score
    df = pd.read_csv(path, sep="\t")
    df = add_q_reformat(df)

    # Preprocess texts
    tokens, lemmas = preprocess_texts(df.q_reformat.tolist())
    df["tokens"], df["lemmas"], df["embedding"] = tokens, lemmas, None

    # Get explanation uids and roles
    exp_uids = []
    exp_roles = []
    exp_idxs = []
    for exp_string in df.explanation.values:
        _uids, _roles = extract_explanation(exp_string)
        uids = []
        roles = []
        idxs = []
        assert len(_uids) == len(_roles)
        for i in range(len(_uids)):
            if _uids[i] not in uid2idx:
                continue
            uids.append(_uids[i])
            roles.append(_roles[i])
            idxs.append(uid2idx[_uids[i]])
        exp_uids.append(uids)
        exp_roles.append(roles)
        exp_idxs.append(idxs)
    df["exp_uids"], df["exp_roles"], df["exp_idxs"] = exp_uids, exp_roles, exp_idxs

    print(df.shape)
    return df
