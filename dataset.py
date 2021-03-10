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
            for document in question_rating["documents"]:
                document_id = document["uuid"]
                document_text = document["docText"]
                relevance = document["relevance"]
                is_gold = document["isGoldWT21"]
                gold_role = document["goldRole"]
                question_ratings.append({"question_id": question_id, "question_text":
                            question_text, "document_id": document_id,
                            "document_text": document_text, "relevance":
                            relevance, "is_gold": is_gold, "gold_role": gold_role})

        df = pd.DataFrame(question_ratings)
        text = df.question_text + " [SEP] " + df.document_text
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

    def get_document(self, document_id):
        documents = self.df.loc[self.df['document_id'] == document_id]
        if len(documents) > 0:
            return documents.loc[0].document_text
        else:
            raise ValueError(f"{document_id} does not exist!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in
                self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item
