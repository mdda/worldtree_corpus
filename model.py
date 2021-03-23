import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
)

from dataset import QuestionRatingDataset, ExplanationDataset, PredictDataset
from retriever import PredictManager, Prediction
from evaluate import mean_average_ndcg


class LinearLayer(nn.Module):
    " batch_norm -> dropout -> linear -> activation "

    def __init__(self, in_feat, out_feat, bn=True, dropout=0.0, activation=None):
        super().__init__()
        layers = []
        if bn:
            layers.append(BatchNorm1dFlat(in_feat))
        if dropout != 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_feat, out_feat))
        if activation is not None:
            layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_layer(x)


class BatchNorm1dFlat(nn.BatchNorm1d):
    "`nn.BatchNorm1d`, but first flattens leading dimensions"

    def forward(self, x):
        if x.dim() == 2:
            return super().forward(x)
        *f, c = x.shape
        x = x.contiguous().view(-1, c)
        return super().forward(x).view(*f, c)


class TransformerRanker(pl.LightningModule):
    def __init__(self, learning_rate=5e-5, num_labels=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=self.num_labels
        )

    def freeze_transformer(
        self,
        freeze_pos=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
    ):
        for name, p in self.transformer.bert.named_parameters():
            name = name.lower()
            if "ln" in name or "layernorm" in name:
                p.requires_grad = not freeze_ln
            elif "wpe" in name or "position_embeddings" in name:
                p.requires_grad = not freeze_pos
            elif "mlp" in name or "dense" in name:
                p.requires_grad = not freeze_ff
            elif "attn" in name or "attention" in name:
                p.requires_grad = not freeze_attn
            else:
                p.requires_grad = False

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.transformer(x["input_ids"], attention_mask=x["attention_mask"])
        return embedding

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if self.num_labels == 1:
            labels = batch["labels"].float()
        else:
            labels = batch["classes"]
        outputs = self.transformer(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        return {"loss": outputs.loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        return {
            "val_loss": outputs.loss,
            "logits": outputs.logits,
            "question_id": batch["question_id"],
            "explanation_id": batch["explanation_id"],
        }

    def validation_epoch_end(self, outs):
        pred_logits = defaultdict(dict)
        for out in outs:
            for pred in zip(out["question_id"], out["explanation_id"], out["logits"]):
                question_id = pred[0]
                explanation_id = pred[1]
                logits = pred[2]
                pred_logits[question_id][explanation_id] = logits
        preds = []
        for question_id, explanation_logits in pred_logits.items():
            if self.num_labels == 1:
                eids = [
                    k
                    for k, v in sorted(
                        explanation_logits.items(), key=lambda i: torch.max(i[1])
                    )[::-1]
                ]
            else:
                eids = [
                    k
                    for k, v in sorted(
                        explanation_logits.items(),
                        key=lambda i: (torch.argmax(i[1]), torch.max(i[1])),
                        reverse=True,
                    )
                ]
            preds.append(Prediction(qid=question_id, eids=eids))
        if self.trainer.log_dir is None:
            predict_dir = ""
        else:
            predict_dir = self.trainer.log_dir
        dataset = QuestionRatingDataset("data/wt-expert-ratings.dev.json")
        ge = dataset.gold_predictions
        ndcg = mean_average_ndcg(ge, preds, 0)
        self.log('ndcg', ndcg)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "question_id": batch["question_id"],
            "explanation_id": batch["explanation_id"],
        }

    def test_epoch_end(self, outs):
        pred_logits = defaultdict(dict)
        for out in outs:
            for pred in zip(out["question_id"], out["explanation_id"], out["logits"]):
                question_id = pred[0]
                explanation_id = pred[1]
                logits = pred[2]
                pred_logits[question_id][explanation_id] = logits
        preds = []
        for question_id, explanation_logits in pred_logits.items():
            if self.num_labels == 1:
                eids = [
                    k
                    for k, v in sorted(
                        explanation_logits.items(), key=lambda i: torch.max(i[1])
                    )[::-1]
                ]
            else:
                eids = [
                    k
                    for k, v in sorted(
                        explanation_logits.items(),
                        key=lambda i: (torch.argmax(i[1]), torch.max(i[1])),
                        reverse=True,
                    )
                ]
            preds.append(Prediction(qid=question_id, eids=eids))
        if self.trainer.log_dir is None:
            predict_dir = ""
        else:
            predict_dir = self.trainer.log_dir
        dataset = QuestionRatingDataset("data/wt-expert-ratings.dev.json")
        ge = dataset.gold_predictions
        mean_average_ndcg(ge, preds, 0)
        with open(os.path.join(predict_dir, "logits.dev.model.pkl"), "wb") as f:
            pickle.dump(pred_logits, f, pickle.HIGHEST_PROTOCOL)

        PredictManager.write(os.path.join(predict_dir, "predict.dev.model.txt"), preds)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--neg_samples", type=int, default=0)
    args = parser.parse_args()
    if args.num_labels !=1 and args.num_labels != 4:
        raise NotImplementedError("num labels can only either be 1 or 4")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # data
    exp_dataset = ExplanationDataset("data/tables")
    train_dataset = QuestionRatingDataset(
        "data/wt-expert-ratings.train.json",
        explanation_dataset=exp_dataset,
        neg_samples=args.neg_samples,
        tokenizer=tokenizer,
    )
    val_dataset = QuestionRatingDataset(
        "data/wt-expert-ratings.dev.json", tokenizer=tokenizer
    )
    pred_dataset = PredictDataset(
        "predict.dev.baseline-retrieval.hyperopt.txt", tokenizer, val_dataset, exp_dataset
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    pred_dataloader = torch.utils.data.DataLoader(
        pred_dataset, batch_size=args.batch_size, shuffle=False
    )

    # model
    model = TransformerRanker(num_labels=args.num_labels)

    # ------------
    # training
    # ------------
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="ndcg", mode='max', patience=2)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="ndcg", mode='max', verbose=True)
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping_callback, checkpoint_callback]
    )
    trainer.fit(model, train_loader, pred_dataloader)

if __name__ == "__main__":
    cli_main()
