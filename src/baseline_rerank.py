import os
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Set, Callable, Any, Dict

import pytorch_lightning as pl
import torch
from comet_ml import BaseExperiment
from dotenv import load_dotenv
from pydantic import BaseModel
from pytorch_lightning.loggers import CometLogger
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    BatchEncoding,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput, TokenClassifierOutput

from baseline_retrieval import (
    Data,
    Retriever,
    SpacyProcessor,
    Prediction,
    ResultAnalyzer,
)
from dataset import QuestionAnswer, ExplanationUsed
from extra_data import SplitEnum, analyze_lengths
from losses import APLoss
from rankers import StageRanker
from vectorizers import BM25Vectorizer


class Config(BaseModel):
    bs: int = 1
    top_n: int = 128
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 1

    net_features_type: str = "transformer"

    net_ranker_type: str = "rnn"
    net_ranker_num_layers: int = 1
    net_ranker_input_size: int = 768
    net_ranker_hidden_size: int = 128

    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    num_epochs: int = 3
    overfit_pct: float = 0.0

    data_name: str = "textgraphs"
    loss_name = "map"


class Example(BaseModel):
    query: str
    docs: List[str]
    labels: List[int]


class RankerNet(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class RnnRankerNet(RankerNet):
    def __init__(self, config: Config):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=config.net_ranker_input_size,
            hidden_size=config.net_ranker_hidden_size,
            num_layers=config.net_ranker_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(config.net_ranker_hidden_size * 2, 1), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        num, dim = x.shape
        x = x.unsqueeze(dim=0)
        x, states = self.rnn(x)
        x = self.linear(x)
        x = torch.squeeze(x, dim=-1)
        assert tuple(x.shape) == (1, num)
        return x


class TransformerRankerNet(RankerNet):
    def __init__(self, config: Config):
        super().__init__()
        self.transformer: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            config.model_name, num_labels=1
        )

    def forward(self, x: Tensor) -> Tensor:
        num_seq, dim = x.shape
        x = torch.unsqueeze(x, dim=0)
        output: TokenClassifierOutput = self.transformer(
            inputs_embeds=x, return_dict=True
        )
        x = output.logits.squeeze(dim=-1)
        x = torch.sigmoid(x)
        assert tuple(x.shape) == (1, num_seq)
        return x


class FeatureNet(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def forward(self, x: BatchEncoding) -> Tensor:
        raise NotImplementedError


class TransformerFeatureNet(FeatureNet):
    def __init__(self, config: Config):
        super().__init__()
        self.transformer: PreTrainedModel = AutoModel.from_pretrained(config.model_name)

    def forward(self, x: BatchEncoding) -> Tensor:
        num_seq, seq_len = x.input_ids.shape
        outputs: BaseModelOutput = self.transformer(**x, return_dict=True)
        _, _, dim = outputs.last_hidden_state.shape
        x: Tensor = outputs.last_hidden_state[:, 0, :]
        assert tuple(x.shape) == (num_seq, dim)
        return x


class HierarchicalRankerNet(nn.Module):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self, net_features: FeatureNet, net_ranker: RankerNet):
        super().__init__()
        self.net_features = net_features
        self.net_ranker = net_ranker

    def forward(self, x: BatchEncoding) -> Tensor:
        x = self.net_features(x)
        x = self.net_ranker(x)
        return x


def make_net(config: Config) -> nn.Module:
    net_ranker = dict(
        rnn=RnnRankerNet(config), transformer=TransformerRankerNet(config)
    )[config.net_ranker_type]
    net_features = dict(transformer=TransformerFeatureNet(config))[
        config.net_features_type
    ]
    net = HierarchicalRankerNet(net_features, net_ranker)
    return net


def test_net(
    inputs: BatchEncoding = None,
    labels: Tensor = None,
    config=Config(),
    num_seq=128,
    seq_len=40,
):
    if inputs is None:
        x = torch.ones(num_seq, seq_len, dtype=torch.long)
        inputs = BatchEncoding(data=dict(input_ids=x, attention_mask=x))

    if labels is None:
        labels = torch.lt(torch.randn(config.bs, num_seq), 0.05).float()

    net = make_net(config)
    outputs = net(inputs)
    assert tuple(outputs.shape) == (config.bs, num_seq)
    assert outputs.shape == labels.shape
    loss_fn = APLoss()
    loss = loss_fn(outputs, labels)
    print(dict(loss=loss))


class RerankDataset(Dataset):
    """
    Use ranker to get preds
    For each question
        Get top_n preds + gold explanations
        Fix bs=1 to avoid complication
        Target (tensor) shape is (bs, top_n) in {0, 1}
        Input (BatchEncoding) shape is (bs, top_n, seq_len)
    """

    def __init__(
        self,
        data: Data,
        retriever: Retriever,
        tokenizer: PreTrainedTokenizer,
        top_n: int,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.retriever = retriever
        self.top_n = top_n
        self.uid_to_text = {s.uid: s.raw_txt for s in self.data.statements}
        self.examples = self.load()

    @staticmethod
    def qn_to_text(q: QuestionAnswer):
        return q.question.raw_txt + " " + q.answers[0].raw_txt

    @property
    def preds_and_qns(self) -> Tuple[List[Prediction], List[QuestionAnswer]]:
        preds = self.retriever.run(self.data)
        ResultAnalyzer().run_map_loss(self.data, preds, top_n=self.top_n)
        assert len(preds) == len(self.data.questions)
        indices = []
        for i, q in enumerate(self.data.questions):
            if q.explanation_gold:
                indices.append(i)
            else:
                print(dict(qn_no_explanation_gold=q.question_id))
        return [preds[i] for i in indices], [self.data.questions[i] for i in indices]

    def truncate_uids(
        self, p: Prediction, q: QuestionAnswer
    ) -> Tuple[List[str], Set[str]]:
        # Add missing gold uids but also maintain relative ordering
        e: ExplanationUsed
        uids_gold = set()
        for e in q.explanation_gold:
            if e.uid in self.uid_to_text.keys():
                uids_gold.add(e.uid)
            else:
                # Only 1 case in train set: b4d1-4fcf-d14e-cb3b (dev set none)
                print(dict(uid_not_in_uid_to_text=e.uid))
        assert uids_gold

        uids = []
        num_distractor = 0
        for u in p.uids:
            if u in uids_gold:
                uids.append(u)
            elif num_distractor + len(uids_gold) < self.top_n:
                uids.append(u)
                num_distractor += 1

        assert len(uids) == self.top_n
        assert uids_gold.issubset(uids)
        return uids, uids_gold

    def load(self) -> List[Example]:
        preds, qns = self.preds_and_qns
        examples = []
        for p, q in zip(preds, qns):
            uids, uids_gold = self.truncate_uids(p, q)
            examples.append(
                Example(
                    query=self.qn_to_text(q),
                    docs=[self.uid_to_text[u] for u in uids],
                    labels=[int(u in uids_gold) for u in uids],
                )
            )
        return examples

    def make_tokens(self, texts_a: List[str], texts_b) -> BatchEncoding:
        return self.tokenizer(texts_a, texts_b, padding=True, return_tensors="pt")

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def collate_fn(self) -> Callable:
        def fn(examples: List[Example]) -> Tuple[BatchEncoding, Tensor]:
            assert len(examples) == 1
            texts_b = examples[0].docs
            texts_a = [examples[0].query] * len(texts_b)
            x = self.make_tokens(texts_a, texts_b)
            y = Tensor([e.labels for e in examples])
            return x, y

        return fn

    def analyze(self):
        print(type(self).__name__)
        info = dict(
            examples=len(self.examples),
            lengths=analyze_lengths(
                [
                    len(f"{e.query} {doc}".split())
                    for e in self.examples
                    for doc in e.docs
                ]
            ),
            labels=Counter([i for e in self.examples for i in e.labels]),
        )
        print(info)
        print(self.examples[0])
        print(self.collate_fn([self.examples[0]]))


def get_retriever() -> Retriever:
    return Retriever(
        preproc=SpacyProcessor(remove_stopwords=True),
        ranker=StageRanker(num_per_stage=[1, 2, 4, 8, 16], scale=1.25),
        vectorizer=BM25Vectorizer(binary=True, use_idf=True, k1=2.0, b=0.5),
    )  # Dev MAP=0.4861, recall@512=0.9345


def test_dataset(data_split=SplitEnum.dev, config=Config()):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    data = Data(data_split=data_split)
    data.load()
    dataset = RerankDataset(data, get_retriever(), tokenizer, config.top_n)

    # Num workers is not too important: default(0) -> 49 it/s, 4 -> 79 it/s but bottleneck is likely model
    loader = DataLoader(
        dataset, batch_size=config.bs, shuffle=False, collate_fn=dataset.collate_fn,
    )
    limit = 100
    for i, _ in tqdm(enumerate(loader), total=limit):
        if i == limit:
            break

    inputs, labels = next(iter(loader))
    num_seq, seq_len = inputs.input_ids.shape
    assert tuple(labels.shape) == (config.bs, num_seq)
    test_net(inputs=inputs, labels=labels, config=config)
    dataset.analyze()


class System(pl.LightningModule):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = kwargs  # For logging
        self.config = Config(**kwargs)
        self.net = make_net(self.config)
        self.loss_fn = self.make_loss_fn()
        self.ds_train = self.make_dataset(SplitEnum.train)
        self.ds_dev = self.make_dataset(SplitEnum.dev)

    def make_loss_fn(self) -> nn.Module:
        mapping = dict(
            mse=nn.MSELoss(), crossentropy=nn.CrossEntropyLoss(), map=APLoss()
        )
        return mapping[self.config.loss_name]

    def make_dataset(self, data_split: SplitEnum) -> RerankDataset:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        data = Data(data_split=data_split)
        data.load()
        mapping = dict(
            textgraphs=RerankDataset(
                data, get_retriever(), tokenizer, self.config.top_n
            )
        )
        ds = mapping[self.config.data_name]
        return ds

    def setup(self, stage: str):
        if isinstance(self.logger, CometLogger):
            exp: BaseExperiment = self.logger.experiment
            for path in Path(".").glob("*.py"):
                exp.log_asset(str(path), overwrite=True)

    def forward(self, inputs: BatchEncoding) -> Tensor:
        return self.net(inputs)

    def training_step(
        self, batch: Tuple[BatchEncoding, Tensor], i: int
    ) -> Dict[str, Any]:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        log = dict(train_loss=loss)
        return dict(loss=loss, log=log)

    def validation_step(
        self, batch: Tuple[BatchEncoding, Tensor], i: int
    ) -> Dict[str, Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return dict(val_loss=loss)

    # def run_eval(self, data_split: SplitEnum) -> float:
    #     ds = {SplitEnum.train: self.ds_train, SplitEnum.dev: self.ds_dev}[data_split]
    #     retriever = TextPairRetriever(net=self.net, ds=ds)
    #     data = ds.data
    #     preds = retriever.run(data)
    #     qid_to_score = Scorer().run(data.path_gold, preds)
    #     ResultAnalyzer().run(data, preds)
    #     scores = list(qid_to_score.values())
    #     return sum(scores) / len(scores)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = {}
        for d in outputs:
            for k, v in d.items():
                results.setdefault(k, []).append(v)
        log = {k: torch.stack(v).mean().item() for k, v in results.items()}
        # log.update(val_map=self.run_eval(SplitEnum.dev))
        print(log)
        return dict(val_loss=log["val_loss"], log=log)

    @property
    def num_train_steps(self) -> int:
        ds = self.make_dataset(SplitEnum.train)
        return self.config.num_epochs * (len(ds) // self.config.bs)

    def configure_optimizers(self,) -> Tuple[List[Optimizer], List[LambdaLR]]:
        # Reference: https://github.com/huggingface/transformers/blob/dbfe34f2f5413448e09277a730142b1f8e8961cf/src/transformers/trainer.py#L414
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {
                "params": [
                    p
                    for n, p in self.net.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.net.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            grouped_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.num_train_steps,
        )
        return [optimizer], [scheduler]

    def make_loader(self, ds: RerankDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.config.bs,
            collate_fn=ds.collate_fn,
            shuffle=shuffle,
            num_workers=os.cpu_count(),
        )

    def train_dataloader(self) -> DataLoader:
        return self.make_loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.make_loader(self.ds_dev, shuffle=False)


def get_logger(save_dir: str, path_dotenv: str) -> CometLogger:
    load_dotenv(path_dotenv)
    assert os.getenv("COMET_KEY") is not None
    return CometLogger(
        api_key=os.getenv("COMET_KEY"),
        workspace=os.getenv("COMET_WORKSPACE"),  # Optional
        project_name=os.getenv("COMET_PROJECT"),  # Optional
        save_dir=save_dir,
        # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        # experiment_name=input("Enter Experiment Name:"),  # Optional
    )


def run_train(logger: CometLogger):
    config = Config()
    system = System(**config.dict())

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.num_epochs,
        overfit_batches=config.overfit_pct,
        gpus=1,
    )  # precision=16 is not faster on P100
    trainer.fit(system)


def main(save_dir="/tmp/comet_logger", path_dotenv="../excluded/.env"):
    if not Path(save_dir).exists():
        logger = get_logger(save_dir, path_dotenv)
        run_train(logger)


if __name__ == "__main__":
    test_net()
    test_dataset()
    main()
