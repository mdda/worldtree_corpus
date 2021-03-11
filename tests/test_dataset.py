import os

import pandas as pd

from dataset import QuestionRatingDataset, ExplanationDataset
from baseline_tfidf import read_explanations, read_questions


def compare_df(df1, df2):
    return df1.reset_index(drop=True).equals(df2.reset_index(drop=True))


def test_explanations_dataset():
    tables_path = "data/tables"
    expected_explanations_arr = []
    for path, _, files in os.walk(tables_path):
        for file in files:
            expected_explanations_arr += read_explanations(os.path.join(path, file))
    expected_explanations = pd.DataFrame(
        expected_explanations_arr, columns=["explanation_id", "explanation_text"]
    )

    explanation_dataset = ExplanationDataset(tables_path)
    actual_explanations = explanation_dataset.explanations

    assert compare_df(expected_explanations, actual_explanations)


def test_questions_dataset():
    questions_path = "data/wt-expert-ratings.dev.json"
    expected_questions_arr = read_questions(questions_path)
    expected_questions = pd.DataFrame(
        expected_questions_arr, columns=["question_id", "question_text"]
    )

    question_dataset = QuestionRatingDataset(questions_path)
    actual_questions = question_dataset.questions

    assert compare_df(expected_questions, actual_questions)
