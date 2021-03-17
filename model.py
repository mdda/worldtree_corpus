from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AdamW,
)

from dataset import QuestionRatingDataset, ExplanationDataset, PredictDataset
from retriever import PredictManager, Prediction


class TransformerRanker(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.transformer = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=1
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.transformer(x["input_ids"], attention_mask=x["attention_mask"])
        return embedding

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float()
        outputs = self.transformer(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        return {"loss": outputs.loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float()
        outputs = self.transformer(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        return {"val_loss": outputs.loss}

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
                logits = pred[2][0]
                pred_logits[question_id][explanation_id] = logits
        preds = []
        for question_id, explanation_logits in pred_logits.items():
            eids = [
                k
                for k, v in sorted(explanation_logits.items(), key=lambda i: i[1])[::-1]
            ]
            preds.append(Prediction(qid=question_id, eids=eids))
        PredictManager.write("predict.dev.model.txt", preds)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    # data
    train_dataset = QuestionRatingDataset(
        "data/wt-expert-ratings.train.json", tokenizer
    )
    val_dataset = QuestionRatingDataset("data/wt-expert-ratings.dev.json", tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # model
    model = TransformerRanker()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.checkpoint_callback.monitor = "val_loss"
    trainer.checkpoint_callback.verbose = True
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    exp_dataset = ExplanationDataset("data/tables")
    pred_dataset = PredictDataset(
        "predict.dev.baseline-retrieval.txt", tokenizer, val_dataset, exp_dataset
    )
    pred_dataloader = torch.utils.data.DataLoader(
        pred_dataset, batch_size=32, shuffle=False
    )
    trainer.test(model, test_dataloaders=pred_dataloader)


if __name__ == "__main__":
    cli_main()
