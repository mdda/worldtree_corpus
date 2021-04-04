import os
import json
import functools
import random
from collections import defaultdict

import torch
import pandas as pd

from retriever import PredictManager


class QuestionRatingDataset(torch.utils.data.Dataset):
    def __init__(self, path, explanation_dataset=None, tokenizer=None, 
        neg_samples=0, neg_retrievals=None):
        with open(path, "rb") as f:
            questions_file = json.load(f)

        if neg_retrievals is not None:  # Group the retrieval results by qid
            neg_retrieval_by_qid = {
                pred.qid:pred.eids for pred in neg_retrievals # neg_retrieval is a [Prediction]
            }

        question_ratings, qid_to_question_text = [], dict()
        for question_rating in questions_file["rankingProblems"]:
            question_id = question_rating["qid"]
            question_text = question_rating["queryText"].replace("[ANSWER]", "")

            qid_to_question_text[question_id] = question_text

            # rename to explanation
            cur_exp_ids = []
            for explanation in question_rating["documents"]:
                explanation_id = explanation["uuid"]
                explanation_text = explanation["docText"]
                relevance = explanation["relevance"]
                is_gold = explanation["isGoldWT21"]
                gold_role = explanation["goldRole"]
                question_ratings.append(
                    {
                        "question_id": question_id,
                        "question_text": question_text,
                        "explanation_id": explanation_id,
                        "explanation_text": explanation_text,
                        "relevance": relevance,
                        "is_gold": is_gold,
                        "gold_role": gold_role,
                    }
                )
                cur_exp_ids.append(explanation_id)
            if neg_samples > 0:  # This is arbitrary negative samples from the full explanations
                assert explanation_dataset is not None
                exp_df = explanation_dataset.explanations
                len_exp = len(exp_df)
                rand_indexes = [
                    random.randrange(0, len_exp) for i in range(neg_samples)
                ]
                exp_samples = exp_df.iloc[rand_indexes]
                for i, exp in exp_samples.iterrows():
                    if i in cur_exp_ids:
                        continue
                    question_ratings.append(
                        {
                            "question_id": question_id,
                            "question_text": question_text,
                            "explanation_id": i,
                            "explanation_text": exp.explanation_text,
                            "relevance": 0,
                            "is_gold": "0",
                            "gold_role": "",
                        }
                    )
                    cur_exp_ids.append(i)
                    
            # Now do negative samples from the training retrieval set 
            #   since it is most like the dev/test retrieval set we'll be tested on
            if neg_retrievals is not None:
                assert explanation_dataset is not None
                #exp_df = explanation_dataset.explanations
                cur_exp_ids_set = set(cur_exp_ids)
                for eid in neg_retrieval_by_qid.get(question_id, []):
                    if eid not in cur_exp_ids_set:
                        #exp = exp_df['explanation_id']=explanation_id
                        question_ratings.append(
                            {
                                "question_id": question_id,
                                "question_text": question_text,
                                "explanation_id": eid,
                                "explanation_text": explanation_dataset.get_explanation(eid),
                                "relevance": 0,
                                "is_gold": "0",
                                "gold_role": "",
                            }
                        )
                        #cur_exp_ids.append(i) # No need here - no dupes...

        self.qid_to_question_text = qid_to_question_text

        print(f"{len(question_ratings)=}")
        if len(question_ratings)>0:
            df = pd.DataFrame(question_ratings)
            df["text"] = self.concat_question_explanation(
                df.question_text, df.explanation_text
            )
            #self.encodings={}
            if tokenizer:
                self.encodings = tokenizer(df.text.tolist(), padding=True, truncation=True)
            self.labels = df.relevance / max(df.relevance)
            self.classes = (df.relevance + 1) // 2
            self.df = df

    def concat_question_explanation(self, question, explanation):
        return question + " [SEP] " + explanation

    @property
    def questions(self):
        return pd.DataFrame(
            self.df.drop_duplicates("question_id"),
            columns=["question_id", "question_text"],
        ).reset_index(drop=True)

    @property
    def gold_predictions(self):
        gold_preds = defaultdict(dict)
        for index, row in self.df.iterrows():
            gold_preds[row.question_id][row.explanation_id] = row.relevance
        return gold_preds

    #@functools.lru_cache(maxsize=4)
    #def get_question(self, question_id):
    #    questions = self.df.loc[self.df["question_id"] == question_id]
    #    if len(questions) > 0:
    #        return questions.iloc[0].question_text
    #    else:
    #        raise ValueError(f"{question_id} does not exist!")
    def get_question(self, question_id):
        if question_id in self.qid_to_question_text:
            return self.qid_to_question_text[question_id]
        raise ValueError(f"{question_id} does not exist!")

    def get_explanation(self, explanation_id):
        explanations = self.df.loc[self.df["explanation_id"] == explanation_id]
        if len(explanations) > 0:
            return explanations.iloc[0].explanation_text
        else:
            raise ValueError(f"{explanation_id} does not exist!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):  # Will fail without a tokeniser
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}

        item["question_id"] = self.df.loc[i]["question_id"]
        item["explanation_id"] = self.df.loc[i]["explanation_id"]

        item["labels"] = torch.tensor(self.labels[i])
        item["classes"] = torch.tensor(self.classes[i])
        item["relevance"] = torch.tensor(self.df.loc[i]["relevance"])
        return item


class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, question_dataset, explanation_dataset):
        preds = PredictManager.read(path)
        preds_df = []
        for pred in preds:
            for eid in pred.eids:
                preds_df.append(
                    {
                        "question_id": pred.qid,
                        "question_text": question_dataset.get_question(pred.qid),
                        "explanation_id": eid,
                        "explanation_text": explanation_dataset.get_explanation(eid),
                    }
                )
        df = pd.DataFrame(preds_df)
        df["text"] = question_dataset.concat_question_explanation(
            df.question_text, df.explanation_text
        )
        if tokenizer:
            self.encodings = tokenizer(df.text.tolist(), padding=True, truncation=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = {key: torch.tensor(val[i]) for key, val in self.encodings.items()}
        item["question_id"] = self.df.loc[i]["question_id"]
        item["explanation_id"] = self.df.loc[i]["explanation_id"]
        return item


class ExplanationDataset(torch.utils.data.Dataset):
    def __init__(self, path_dir):
        self.path_dir = path_dir
        self.df = None

    @property
    def explanations(self):
        if self.df is not None:
            return self.df
        explanations = []
        for path, _, files in os.walk(self.path_dir):
            for file in files:
                explanations += self._read_explanations(os.path.join(path, file))
        explanations_df = pd.DataFrame(
            explanations, columns=("explanation_id", "explanation_text")
        )
        self.df = explanations_df.set_index("explanation_id")
        return self.df

    def get_explanation(self, explanation_id):
        if self.df is None:
            self.explanations
        return self.df.xs(explanation_id).explanation_text
        # explanations = self.df.loc[self.df["explanation_id"] == explanation_id]
        # if len(explanations) > 0:
        #     return explanations.iloc[0].explanation_text
        # else:
        #     raise ValueError(f"{explanation_id} does not exist!")

    @staticmethod
    def _read_explanations(path):
        header = []
        uid = None

        df = pd.read_csv(path, sep="\t", dtype=str)

        for name in df.columns:
            if name.startswith("[SKIP]"):
                if "UID" in name and not uid:
                    uid = name
            else:
                header.append(name)

        if not uid or len(df) == 0:
            warnings.warn("Possibly misformatted file: " + path)
            return []

        return df.apply(
            lambda r: (
                r[uid],
                " ".join(str(s) for s in list(r[header]) if not pd.isna(s)),
            ),
            1,
        ).tolist()
