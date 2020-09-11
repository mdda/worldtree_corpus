import warnings
from pathlib import Path

import fire
import pandas as pd

from preproc import get_questions, get_df_explanations
from rank import get_ranks, get_preds, write_preds, run_scoring

SEP = "#" * 100 + "\n"
MODE_TRAIN = "train"
MODE_DEV   = "dev"
MODE_TEST  = "test"


def get_path_q(path_data: Path, mode: str) -> Path:
    return path_data / f"questions.{mode}.tsv"


def get_path_predict(mode: str) -> str:
    f = {
        MODE_TRAIN: "predict.train.txt",
        MODE_DEV:   "predict.dev.txt",
        MODE_TEST:  "predict.test.txt",
    }[mode]
    print(f"Mode: {mode}, predict file will be saved to: {f}")
    return f


def process_qn(
    path_data: Path, df_exp: pd.DataFrame, use_recursive_tfidf: bool, mode: str,
) -> None:
    print(SEP, f"Processsing {mode} data set")
    uid2idx = {uid: idx for idx, uid in enumerate(df_exp.uid.tolist())}
    path_q = get_path_q(path_data, mode)
    df = get_questions(str(path_q), uid2idx)
    path_predict = get_path_predict(mode)

    print(SEP, "BERT output dir not provided, running TFIDF-only methods")
    print(SEP, "Ranking")
    ranks = get_ranks(df, df_exp, use_recursive_tfidf=use_recursive_tfidf)
    preds = get_preds(ranks, df, df_exp)

    print(SEP, "Writing predictions file")
    write_preds(preds, path_predict)

    if mode in [MODE_TRAIN, MODE_DEV]:
        print(SEP, "Scoring")
        _ = run_scoring(path_q, path_predict)


def main(
    path_data: str = "../tg2020task",
    recurse_tfidf: bool = True,
    do_train: bool = True,
    do_dev: bool = True,
    do_test: bool = True,
) -> None:
    """
  Runs main ranking pipeline, scoring and results analysis (BERT re-ranking not included)
  :param path_data: Normally is "worldtree_corpus_textgraphs2019sharedtask_withgraphvis"
  :param recurse_tfidf: Use iterative TFIDF algo, MAP +0.03
  :param do_train: Run on train set
  :param do_dev: Run on dev set
  :param do_test: Run on test set, no scoring or analysis will be run
  :return: None
  """
    assert do_train or do_dev or do_test, "Can't do nothing!"
    path_data = Path(path_data)
    path_tables = path_data / "tables"

    print(SEP, "Preprocessing")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df_exp = get_df_explanations(path_tables)

    def _process_qn(mode: str) -> None:
        process_qn(path_data, df_exp, recurse_tfidf, mode)

    if do_train:
        _process_qn(MODE_TRAIN)
    if do_dev:
        _process_qn(MODE_DEV)
    if do_test:
        _process_qn(MODE_TEST)


if "__main__" == __name__:
    fire.Fire(main)
