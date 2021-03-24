import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

#os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import (
    #AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
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
    def __init__(self, learning_rate=5e-5, num_labels=1, 
        base="bert",
        loss_style=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_labels = num_labels

        #self.transformer = AutoModelForSequenceClassification.from_pretrained(
        #    bert, num_labels=self.num_labels
        #)

        # https://github.com/PyTorchLightning/pytorch-lightning/issues/3096
        self.transformer = None
        if base=='bert':
            self.transformer = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
                num_labels=self.num_labels)
        if base=='distilbert':
            self.transformer = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                num_labels=self.num_labels)
        if self.transformer is None:
            print(f"TransformerFromInitialised FAILURE : {base}")

        self.fold_testing='dev'  # The default
        self.loss_style=loss_style

        if self.loss_style!=1:
            self.sigmoid = torch.sigmoid
            self.sm = torch.nn.Softmax(dim=-1)
            #self.output_weight = torch.nn.Parameter(
            #    torch.Tensor([0., 1., 1., 1., 1., 1., 1.,]), 
            #    requires_grad=False,
            #)
            self.cc_loss = torch.nn.CrossEntropyLoss(
                                #weight=self.output_weight, 
                                reduce=False,
                                ignore_index=-1,
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

    def loss(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if self.loss_style==1:  # This is the default 'vivek' way
            if self.num_labels == 1:
                labels = batch["labels"].float()
            else:
                labels = batch["classes"]
            outputs = self.transformer(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            return outputs.loss
        # else...

        relevance = batch["relevance"]  # An integer from 0+{1,2,3,4,5,6}
        outputs = self.transformer(
            input_ids, attention_mask=attention_mask, #labels=labels
        )

        target_relevance = relevance
        target_weight = (relevance[:]>0).to(torch.float32)

        output_relevance = outputs.logits[:,0]
        #output_dist      = outputs.logits # Includes relevance[:,0], which is eliminated by 'weight' above
        output_dist      = outputs.logits[:,1:]

        loss_relevance = torch.nn.functional.binary_cross_entropy_with_logits(
            output_relevance, target_weight
        )
        #loss_dist = (target_weight*self.cc_loss(output_dist, target_relevance)).mean()
        loss_dist = (
            target_weight     *self.cc_loss(output_dist, target_relevance-1)
            #torch.nn.CrossEntropyLoss(ignore_index=-1)( torch.tensor([0.,0,0,0,0,0]).unsqueeze(0), torch.tensor([-1]) )
            #+(1.-target_weight)*1.7918 # This is the 'uniform prior' 
            # No need : This is switched based on targets, not the output choice
        ).mean()

        loss = loss_relevance + loss_dist
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        #loss = self.loss(batch)
        #return {"val_loss": loss}
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outs):
        preds, pred_logits = self.get_preds(outs)

        dataset = QuestionRatingDataset(f"data/wt-expert-ratings.dev.json")
        ge = dataset.gold_predictions
        ndcg = mean_average_ndcg(ge, preds, 0, oracle=False)

        self.log_dict({'ndcg': ndcg})
        #return {"ndcg": -ndcg}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        return {
            #"loss": outputs.loss,
            "logits": outputs.logits,
            "question_id": batch["question_id"],
            "explanation_id": batch["explanation_id"],
        }

    def get_preds(self, outs):
        print(f"{self.loss_style=}")
        if self.loss_style==1:  # This is the default 'vivek' way
            return self.get_preds_regular(outs)
        # else
        return self.get_preds_switching(outs)

    def get_pred_logits(self, outs):
        pred_logits = defaultdict(dict)
        for out in outs:
            for pred in zip(out["question_id"], out["explanation_id"], out["logits"]):
                question_id = pred[0]
                explanation_id = pred[1]
                pred_logits[question_id][explanation_id] = pred[2]
        return pred_logits

    def get_preds_regular(self, outs):
        preds, pred_logits = [], self.get_pred_logits(outs)

        if self.num_labels == 1:
            sorter = lambda i: torch.max(i[1])  # Just sort them in score-regression order
        else:
            sorter = lambda i: (torch.argmax(i[1]), torch.max(i[1]))  # Sort by best guess, then magnitude of that guess

        for question_id, explanation_logits in pred_logits.items():
            eids = [ k for k, v in sorted( explanation_logits.items(),
                                           key=sorter, reverse=True, ) ]
            preds.append(Prediction(qid=question_id, eids=eids))
        return preds, pred_logits

    def get_preds_switching(self, outs, deterministic=True):
        preds, pred_logits = [], self.get_pred_logits(outs)
        self.hurdle=0.5
        #self.hurdle=0.25

        sorter=None
        if deterministic:
            # For all the logits, assume that those >0.5 are 'on'
            #   And use the distributions as above
            #     Sort by (in vs out), then best guess, then magnitude of that guess
            sorter = lambda i: (int( i[1]>self.hurdle ), torch.argmax(i[2]), torch.max(i[2]))  

        for question_id, explanation_logits in pred_logits.items():
            # key, relevance_prob, relevance_dist
            explanations = [ (k, self.sigmoid(v[0]), self.sm( v[1:] ))  
                for k, v in explanation_logits.items() ]
            
            #for e in explanations[::40]: # 5 entries each
            for e in explanations[::40]: # 5 entries each
                #   EARLY #('2a54-5bc0-92f5-2816', tensor(0.4961, device='cuda:0'), tensor([0.1379, 0.1735, 0.1838, 0.1733, 0.1752, 0.1563], device='cuda:0'))
                #   LATER #('5003-5988-8e84-ea8d', tensor(0.9974, device='cuda:0'), tensor([0.0016, 0.0169, 0.0998, 0.8352, 0.0380, 0.0084], device='cuda:0'))
                # version_42 : examples of distribution idea
                # f992-5698-76aa-c6de : 0.1424 [0.3784, 0.3788, 0.0781, 0.1283, 0.0143, 0.0220]
                # 20e2-689d-b7c6-528a : 0.9689 [0.2940, 0.4254, 0.2266, 0.0511, 0.0024, 0.0005]
                # b429-cf83-df90-a688 : 0.9995 [0.0009, 0.0028, 0.0065, 0.0047, 0.0110, 0.9740]
                # unfortunately, score is poor...
                s = ', '.join(f'{v:.4f}' for v in e[2])
                print(f"{e[0]} : {e[1].item():.4f} [{s}]")

            if deterministic:
                # See above for elegant sorter
                eids = [ k for k, v_prob, v_dist in sorted( explanations,
                                            key=sorter, reverse=True, ) ]

            preds.append(Prediction(qid=question_id, eids=eids))
            
        return preds, pred_logits

    def test_epoch_end(self, outs):
        preds, pred_logits = self.get_preds(outs)

        # Write out predictions for this fold
        predict_dir = "" if self.trainer.log_dir is None else self.trainer.log_dir
        PredictManager.write(os.path.join(predict_dir, 
            f"predict.{self.fold_testing}.bert0.txt"), preds)
        with open(os.path.join(predict_dir, f"logits.{self.fold_testing}.model.pkl"), "wb") as f:
            pickle.dump(pred_logits, f, pickle.HIGHEST_PROTOCOL)
        if self.fold_testing=='test':
            return # Don't know gold results - nothing to return

        dataset = QuestionRatingDataset(f"data/wt-expert-ratings.{self.fold_testing}.json")
        ge = dataset.gold_predictions
        mean_average_ndcg(ge, preds, 0)
        return 

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def cli_main():
    #pl.seed_everything(1234)
    pl.seed_everything(42)

    # args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--loss_style", type=int, default=1)
    parser.add_argument("--neg_samples", type=int, default=0)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--base", type=str, default="bert")
    parser.add_argument("--fold", type=str, default="dev")
    args = parser.parse_args()
    if args.loss_style==1 and args.num_labels !=1 and args.num_labels != 4:
        raise NotImplementedError("num labels can only either be 1 or 4")
    if args.loss_style==2 and args.num_labels != 6+1:
        raise NotImplementedError("num labels can only 6+1")

    tokenizer = None
    if args.base=='bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if args.base=='distilbert':
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

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
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4,
    )

    #early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=2)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="ndcg", patience=2, mode='max')
    #ndcg
    #checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", verbose=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="ndcg", verbose=True, mode='max')
    trainer = pl.Trainer.from_argparse_args(
        args, 
        gpus=1, 
        precision=16,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=5,
    )

    if args.load is None:
        model = TransformerRanker(num_labels=args.num_labels, 
                                    base=args.base,
                                    loss_style=args.loss_style,
                                 )
        # ------------
        # training
        # ------------
        trainer.fit(model, train_loader, val_loader)
    else:
        model=TransformerRanker.load_from_checkpoint(args.load)
        print("Model Loaded")

    # ------------
    # re-validation or testing
    # ------------
    if args.fold=='dev':
        pred_dataset = PredictDataset(
            "predictions/predict.dev.baseline-retrieval.hyperopt.txt", 
            tokenizer, 
            val_dataset, 
            exp_dataset
        )
    else:
        model.fold_testing='test'
        test_dataset = QuestionRatingDataset(
            "data/wt-expert-ratings.test.json", tokenizer=tokenizer
        )
        pred_dataset = PredictDataset(
            "predictions/predict.test.baseline-retrieval.hyperopt.txt", 
            tokenizer, 
            test_dataset, # different
            exp_dataset,
        )

    pred_dataloader = torch.utils.data.DataLoader(
        pred_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4,
    )
    #model.eval()  # not actually needed, it seems
    print("StartTest")
    trainer.test(model, test_dataloaders=pred_dataloader)

    #if args.fast_dev_run > 0:
    #else:
    #    # automatically choose best model
    #    trainer.test(test_dataloaders=pred_dataloader)


if __name__ == "__main__":
    cli_main()
