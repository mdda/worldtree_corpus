import json

import torch
import pandas as pd

class QuestionRatings(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        with open(path, "rb") as f:
            questions_file = json.load(f)
        question_ratings = []

        for question_rating in questions_file["rankingProblems"]:
            question_id = question_rating["qid"]
            question_text = question_rating["queryText"]
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
        self.encodings = tokenizer(text.tolist(), padding=True,
                                   truncation=True)
        self.labels = df.is_gold.astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in
                self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item
