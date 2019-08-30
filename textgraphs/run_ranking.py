import warnings
from pathlib import Path

import fire
import pandas as pd
from scipy import stats

from evaluate_with_role_breakdown import ListShouldBeEmptyWarning
from preproc import get_questions, get_df_explanations
from rank import get_ranks, get_preds, write_preds, run_scoring
from rerank import make_score_data, preproc_trn_data, read_predict_txt, rerank_ranks, read_preds

SEP = "#" * 100 + "\n"
MODE_TRAIN = "train"
MODE_DEV = "dev"
MODE_TEST = "test"


def get_path_q(path_data: Path, mode: str) -> Path:
  file_q = {
      MODE_TRAIN: "ARC-Elementary+EXPL-Train.tsv",
      MODE_DEV: "ARC-Elementary+EXPL-Dev.tsv",
      MODE_TEST: "ARC-Elementary+EXPL-Test-Masked.tsv",
  }[mode]
  return path_data.joinpath("questions", file_q)


def get_path_predict(mode: str) -> str:
  f = {
      MODE_TRAIN: "predict_trn.txt",
      MODE_DEV: "predict_dev.txt",
      MODE_TEST: "predict.txt",
  }[mode]
  print(f"Mode: {mode}, predict file will be saved to: {f}")
  return f


def get_path_df_scores(mode: str, clean_trn: bool = False) -> str:
  path = f"df_scores_{mode}.csv"
  if clean_trn:
    # For training, the train data will be shuffled and truncated
    # However, there is the option to have a "clean" version for inference
    path = "unshuffled_undropped_" + path
  return path


def get_path_results(output_dir: str, mode: str) -> str:
  if mode == MODE_DEV:
    mode = "eval"  # Slightly different naming convention in BERT codebase
  return f"{output_dir}/{mode}_results.tsv"


def process_qn(
    path_data: Path,
    df_exp: pd.DataFrame,
    use_recursive_tfidf: bool,
    show_analysis: bool,
    bert_output_dir: str,
    mode: str,
    plot_name: str = "",
    do_rerank_by_role: bool = False,
    do_average_ranks: bool = True,
) -> None:
  print(SEP, f"Processsing {mode} data set")
  uid2idx = {uid: idx for idx, uid in enumerate(df_exp.uid.tolist())}
  path_q = get_path_q(path_data, mode)
  df = get_questions(str(path_q), uid2idx, path_data)
  path_predict = get_path_predict(mode)

  if bert_output_dir == "":
    print(SEP, "BERT output dir not provided, running TFIDF-only methods")
    print(SEP, "Ranking")
    ranks = get_ranks(
        df,
        df_exp,
        use_embed=False,
        use_recursive_tfidf=use_recursive_tfidf,
    )
    preds = get_preds(ranks, df, df_exp)

    print(SEP, "Writing predictions file")
    write_preds(preds, path_predict)
    prepare_rerank_data(df, df_exp, ranks, mode)
  else:
    print(SEP, "BERT output dir provided, running re-ranking pipeline")
    path_predict = do_rerank(df, df_exp, bert_output_dir, mode,
                             do_average_ranks)

  if mode in [MODE_TRAIN, MODE_DEV]:
    print(SEP, "Scoring")
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=ListShouldBeEmptyWarning)
      qid2score = run_scoring(path_q, path_predict)

def do_rerank(
    df: pd.DataFrame,
    df_exp: pd.DataFrame,
    bert_output_dir: str,
    mode: str,
    do_average_ranks: bool = True,
) -> str:
  path_predict = get_path_predict(mode)
  ranks = read_predict_txt(df, df_exp, path_predict)
  path_df_scores = get_path_df_scores(mode, clean_trn=(mode == MODE_TRAIN))
  df_scores = pd.read_csv(path_df_scores)
  df_scores["pred"] = read_preds(get_path_results(bert_output_dir, mode))
  if mode != MODE_TEST:
    spearman = stats.spearmanr(df_scores.score, df_scores.pred)
    print("Spearman score:", spearman)
  reranks = rerank_ranks(df, ranks, df_scores.pred.values, do_average_ranks)
  path_predict_rerank = "rerank_" + path_predict
  write_preds(get_preds(reranks, df, df_exp), path_predict_rerank)
  print("Reranked predictions saved to:", path_predict_rerank)
  return path_predict_rerank


def prepare_rerank_data(
    df: pd.DataFrame,
    df_exp: pd.DataFrame,
    ranks: list,
    mode: str,
) -> None:
  path_df_scores = get_path_df_scores(mode)

  if mode == MODE_TRAIN:
    df_scores = make_score_data(df, df_exp, ranks)
    df_scores.to_csv(get_path_df_scores(mode, clean_trn=True), index=False)
    df_scores = preproc_trn_data(df_scores)
  else:
    # df_scores = make_score_data(df, df_exp, ranks, top_n=1024)
    df_scores = make_score_data(df, df_exp, ranks)
  print(SEP, "Preparing rerank data")
  print("Saving rerank data to:", path_df_scores)
  df_scores.to_csv(path_df_scores, index=False)


def main(
    path_data: str,
    explanation_combos: bool = True,
    recurse_tfidf: bool = True,
    do_train: bool = True,
    do_dev: bool = True,
    do_test: bool = True,
    bert_output_dir: str = "",
    show_analysis: bool = True,
    plot_name: str = "",
    do_rerank_by_role: bool = False,
    do_average_ranks: bool = True,
) -> None:
  """
  Runs main ranking pipeline, scoring and results analysis (BERT re-ranking not included)
  :param path_data: Normally is "worldtree_corpus_textgraphs2019sharedtask_withgraphvis"
  :param explanation_combos: Explanations with combos eg man is human;male -> separate entries, MAP +0.01
  :param recurse_tfidf: Use iterative TFIDF algo, MAP +0.03
  :param do_train: Run on train set
  :param do_dev: Run on dev set
  :param do_test: Run on test set, no scoring or analysis will be run
  :param show_analysis: Run analysis on results
  :param bert_output_dir: Cloud bucket path for BERT output model directory
  :param do_rerank_by_role: Rebalance ranking to prioritize miniority role classes, MAP -0.03
  :param do_average_ranks: Average old and new ranks when reranking, MAP +0.01
  :param plot_name: Unique identifier to save results dataframe .csv file
  :return: None
  """
  assert do_train or do_dev or do_test, "Can't do nothing!"
  path_data = Path(path_data)
  path_tables = path_data.joinpath(
      "annotation/expl-tablestore-export-2017-08-25-230344/tables")

  print(SEP, "Preprocessing")
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    df_exp = get_df_explanations(
        path_tables,
        explanation_combos,
        path_data,
    )

  def _process_qn(mode: str) -> None:
    process_qn(
        path_data,
        df_exp,
        recurse_tfidf,
        show_analysis,
        bert_output_dir,
        mode=mode,
        plot_name=plot_name,
        do_rerank_by_role=do_rerank_by_role,
        do_average_ranks=do_average_ranks,
    )

  if do_train:
    _process_qn(MODE_TRAIN)
  if do_dev:
    _process_qn(MODE_DEV)
  if do_test:
    _process_qn(MODE_TEST)


if "__main__" == __name__:
  fire.Fire(main)
