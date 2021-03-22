import os
import json

import torch
import torch.nn.functional as F
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AdamW,
)

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from dataset import QuestionRatingDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1
)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model.to(device)
model.train()

train_dataset = QuestionRatingDataset("data/wt-expert-ratings.train.json", tokenizer=tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(5):
    overall_loss = 0
    iter = 1e-8
    for batch in tqdm(train_loader, desc=f"{epoch+1}/5: "):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float())
        loss = outputs.loss
        loss.backward()
        overall_loss += loss.detach().cpu()
        iter += 1
        optimizer.step()
    print(overall_loss / iter)

model.eval()

torch.save(model.state_dict(), "model")
model.load_state_dict(torch.load("model"))
model.eval()


def read_explanations(path):
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
        lambda r: (r[uid], " ".join(str(s) for s in list(r[header]) if not pd.isna(s))),
        1,
    ).tolist()


def read_questions(path):
    questions_list = []

    with open(path, "rb") as f:
        questions_file = json.load(f)

    for ranking_problem in questions_file["rankingProblems"]:
        question_id = ranking_problem["qid"]
        question_text = ranking_problem["queryText"].replace("[ANSWER]", "")
        questions_list.append((question_id, question_text))

    return questions_list


explanations = []
for path, _, files in os.walk("data/tables"):
    for file in files:
        explanations += read_explanations(os.path.join(path, file))

if not explanations:
    warnings.warn("Empty explanations")

questions = read_questions("data/wt-expert-ratings.dev.json")

df_q = pd.DataFrame(questions, columns=("qid", "question"))
df_e = pd.DataFrame(explanations, columns=("uid", "text"))

vectorizer = TfidfVectorizer().fit(df_q["question"]).fit(df_e["text"])
X_q = vectorizer.transform(df_q["question"])
X_e = vectorizer.transform(df_e["text"])
X_dist = cosine_distances(X_q, X_e)

for i_question, distances in tqdm(
    enumerate(X_dist), desc="data/wt-expert-ratings.dev.json", total=X_q.shape[0]
):
    logits = []
    for i_explanation in np.argsort(distances)[:100]:
        text = (
            df_q.loc[i_question]["question"]
            + " [SEP] "
            + df_e.loc[i_explanation]["text"]
        )
        encodings = tokenizer(text, padding=True, truncation=True)
        input_ids = torch.tensor(encodings["input_ids"]).view(1, -1).to(device)
        attention_mask = (
            torch.tensor(encodings["attention_mask"]).view(1, -1).to(device)
        )
        outputs = model(input_ids, attention_mask=attention_mask)
        logits.append(outputs.logits[0][0].detach().cpu())
    for i_explanation in np.argsort(distances)[:100][np.argsort(logits)[::-1]]:
        print(
            "{}\t{}".format(df_q.loc[i_question]["qid"], df_e.loc[i_explanation]["uid"])
        )
