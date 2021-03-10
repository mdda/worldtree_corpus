import json

import torch
import pandas as pd

class QuestionRatings(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer=None):
        with open(path, "rb") as f:
            questions_file = json.load(f)
        question_ratings = []

        for question_rating in questions_file["rankingProblems"]:
            question_id = question_rating["qid"]
            question_text = question_rating["queryText"]
            # rename to explanation
            for explanation in question_rating["documents"]:
                explanation_id = explanation["uuid"]
                explanation_text = explanation["docText"]
                relevance = explanation["relevance"]
                is_gold = explanation["isGoldWT21"]
                gold_role = explanation["goldRole"]
                question_ratings.append({"question_id": question_id, "question_text":
                            question_text, "explanation_id": explanation_id,
                            "explanation_text": explanation_text, "relevance":
                            relevance, "is_gold": is_gold, "gold_role": gold_role})

        df = pd.DataFrame(question_ratings)
        text = df.question_text + " [SEP] " + df.explanation_text
        self.df = df
        if tokenizer:
            self.encodings = tokenizer(text.tolist(), padding=True,
                                    truncation=True)
        self.labels = df.relevance/max(df.relevance)

    def get_question(self, question_id):
        questions = self.df.loc[self.df['question_id'] == question_id]
        if len(questions) > 0:
            return questions.loc[0].question_text
        else:
            raise ValueError(f"{question_id} does not exist!")

    def get_explanation(self, explanation_id):
        explanations = self.df.loc[self.df['explanation_id'] == explanation_id]
        if len(explanations) > 0:
            return explanations.loc[0].explanation_text
        else:
            raise ValueError(f"{explanation_id} does not exist!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in
                self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item
