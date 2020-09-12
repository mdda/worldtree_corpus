import os
import random
from collections import Counter
from pathlib import Path
from typing import Tuple, Any, List, Callable, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from comet_ml.experiment import BaseExperiment
from dotenv import load_dotenv
from pydantic import BaseModel
from pytorch_lightning.loggers import CometLogger
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput

from baseline_retrieval import Data as TextGraphsData, Retriever, Scorer, ResultAnalyzer
from dataset import QuestionAnswer, ExplanationUsed
from extra_data import SplitEnum, analyze_lengths

TextPairInput = Tuple[BatchEncoding, BatchEncoding]
TextPairBatch = Tuple[TextPairInput, Tensor]


class TextPairConfig(BaseModel):
    bs: int = 32
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    p_dropout: float = 0.2
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    num_epochs: int = 3
    overfit_pct: float = 0.0
    data_name: str = "textgraphs"
    loss_name = "crossentropy"


def is_tpu_available() -> bool:
    """
    TPU Training gets error
    E tensorflow/compiler/xla/xla_client/tf_logging.cc:11] Failed to retrieve
    mesh configuration: Connection reset by peer (14)
    """
    return os.getenv("COLAB_TPU_ADDR") is not None


def get_device_kwargs() -> Dict[str, int]:
    if is_tpu_available():
        kwargs = dict(tpu_cores=8)
    else:
        assert torch.cuda.is_available()
        kwargs = dict(gpus=1)
    print(dict(device_kwargs=kwargs))
    return kwargs


class TextPairNet(nn.Module):
    def __init__(self, model: PreTrainedModel, num_labels: int, p_dropout: float):
        super().__init__()
        self.model = model

        dim = model.config.dim
        # nn.Bilinear is really expensive! dim**3 -> 1.7 GB bigger checkpoint
        self.linear = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(p_dropout)
        self.final = nn.Linear(dim, num_labels)
        self.activation = nn.ReLU()

    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def embed_texts(self, inputs: BatchEncoding) -> Tensor:
        num_seq, seq_len = inputs.input_ids.shape
        outputs: BaseModelOutput = self.model(
            **inputs, return_dict=True,
        )
        _, _, dim = outputs.last_hidden_state.shape
        pooled: Tensor = outputs.last_hidden_state[:, 0, :]
        assert tuple(pooled.shape) == (num_seq, dim)
        return pooled

    def fuse_embeds(self, a: Tensor, b: Tensor) -> Tensor:
        assert a.shape == b.shape
        x = torch.cat([a, b], dim=-1)
        x = self.linear(x)
        assert x.shape == a.shape
        return x

    def run_head(self, x: Tensor) -> Tensor:
        x = self.activation(x)
        x = self.dropout(x)
        x = self.final(x)
        return x

    def forward(self, inputs: TextPairInput) -> Tensor:
        a = self.embed_texts(inputs[0])
        b = self.embed_texts(inputs[1])
        x = self.fuse_embeds(a, b)
        x = self.run_head(x)
        return x


def test_net(
    inputs: TextPairInput = None, config=TextPairConfig(), max_seq_len=128,
):
    if inputs is None:
        x = torch.ones(config.bs, max_seq_len, dtype=torch.long)
        a = BatchEncoding(data=dict(input_ids=x, attention_mask=x))
        b = BatchEncoding(data=dict(input_ids=x, attention_mask=x))
        inputs = (a, b)

    model = TextPairNet(
        model=AutoModel.from_pretrained(config.model_name),
        num_labels=config.num_labels,
        p_dropout=config.p_dropout,
    )
    outputs = model(inputs)
    assert tuple(outputs.shape) == (config.bs, config.num_labels)


class QnAnsExplanationsRetriever(Retriever):
    def get_gold_explains(self, q: QuestionAnswer, data: TextGraphsData) -> List[str]:
        e: ExplanationUsed
        explains = [
            self.preproc.run(s)
            for e in q.explanation_gold
            for s in data.uid_to_statements[e.uid]
        ]
        assert explains
        return explains

    def make_query(self, q: QuestionAnswer, data: TextGraphsData) -> str:
        texts = [self.preproc.run(q.question), self.preproc.run(q.answers[0])]
        texts.extend(self.get_gold_explains(q, data))
        return " ".join(texts)


class TextPairExample(BaseModel):
    query: str
    fact: str
    label: bool


class TextPairDataset(Dataset):
    def __getitem__(self, i: int) -> TextPairExample:
        raise NotImplementedError

    @property
    def num_labels(self) -> int:
        raise NotImplementedError

    @property
    def collate_fn(self) -> Callable:
        raise NotImplementedError


class TextGraphsQueryFactDataset(TextPairDataset):
    def __init__(
        self, data_split: SplitEnum, tokenizer: PreTrainedTokenizer, top_n=64,
    ):
        self.tokenizer = tokenizer
        self.top_n = top_n
        self.data = TextGraphsData(data_split=data_split)
        self.retriever = QnAnsExplanationsRetriever()
        self.label_map = {False: 0, True: 1}
        self.examples = self.load()

    @property
    def num_labels(self) -> int:
        return len(self.label_map)

    def load(self) -> List[TextPairExample]:
        r = self.retriever
        data = self.data
        data.load()
        questions = [q for q in data.questions if q.explanation_gold]
        # Train set has 1 question without explanations: Mercury_7221305

        facts: List[str] = [r.preproc.run(s) for s in data.statements]
        queries: List[str] = [r.make_query(q, data) for q in questions]
        ranking = r.rank(queries, facts)
        assert len(ranking == len(questions))

        examples: List[TextPairExample] = []
        for q, rank in tqdm(zip(questions, ranking)):
            q_text = r.preproc.run(q.question) + " " + r.preproc.run(q.answers[0])

            facts_gold = set(r.get_gold_explains(q, data))
            for f in facts_gold:
                examples.append(TextPairExample(query=q_text, fact=f, label=True))

            facts_related = set()
            for i in rank:
                if facts[i] not in facts_gold:
                    facts_related.add(facts[i])
                if len(facts_related) == len(facts_gold):
                    break

            for f in facts_related:
                examples.append(TextPairExample(query=q_text, fact=f, label=False))

        return examples

    def make_tokens(self, texts: List[str]) -> BatchEncoding:
        return self.tokenizer(texts, padding=True, return_tensors="pt")

    def __getitem__(self, i: int) -> TextPairExample:
        return self.examples[i]

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def collate_fn(self) -> Callable:
        def fn(examples: List[TextPairExample]) -> TextPairBatch:
            a = self.make_tokens([e.query for e in examples])
            b = self.make_tokens([e.fact for e in examples])
            labels = torch.Tensor([self.label_map[e.label] for e in examples]).long()
            return (a, b), labels

        return fn

    def analyze(self, k=5):
        print(type(self).__name__)
        query_lengths, fact_lengths = [], []
        for e in self.examples:
            query_lengths.append(len(e.query.split()))
            fact_lengths.append(len(e.fact.split()))
        info = dict(
            num_examples=len(self),
            query_lengths=analyze_lengths(query_lengths),
            fact_lengths=analyze_lengths(fact_lengths),
            labels=Counter([e.label for e in self.examples]),
        )
        print(info)
        random.seed(42)
        indices = random.sample(range(len(self)), k=k)
        for i in indices:
            print(self[i])


def test_dataset(data_split=SplitEnum.train, config=TextPairConfig()):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = TextGraphsQueryFactDataset(data_split, tokenizer)
    dataset.analyze()

    # Num workers is not too important: default(0) -> 49 it/s, 4 -> 79 it/s but bottleneck is likely model
    loader = DataLoader(
        dataset, batch_size=config.bs, shuffle=False, collate_fn=dataset.collate_fn,
    )
    limit = 100
    for i, _ in tqdm(enumerate(loader), total=limit):
        if i == limit:
            break

    inputs, labels = next(iter(loader))
    test_net(inputs=inputs, config=config)


def score_acc(logits: Tensor, y: Tensor) -> Tensor:
    preds = torch.argmax(logits, dim=-1)
    return preds.eq(y).float().mean()


class TextPairSystem(pl.LightningModule):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = kwargs  # For logging
        self.config = TextPairConfig(**kwargs)
        self.net = self.make_net()
        self.loss_fn = self.make_loss_fn()

    def make_net(self) -> TextPairNet:
        transformer = AutoModel.from_pretrained(self.config.model_name)
        return TextPairNet(transformer, self.config.num_labels, self.config.p_dropout)

    def make_loss_fn(self) -> nn.Module:
        mapping = dict(crossentropy=nn.CrossEntropyLoss())
        return mapping[self.config.loss_name]

    def make_dataset(self, data_split: SplitEnum) -> TextPairDataset:
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        mapping = dict(textgraphs=TextGraphsQueryFactDataset(data_split, tokenizer))
        ds = mapping[self.config.data_name]
        assert ds.num_labels == self.config.num_labels
        return ds

    def setup(self, stage: str):
        if isinstance(self.logger, CometLogger):
            exp: BaseExperiment = self.logger.experiment
            for path in Path(".").glob("*.py"):
                exp.log_asset(str(path), overwrite=True)

    def forward(self, inputs: TextPairInput) -> Tensor:
        return self.net(inputs)

    def training_step(self, batch: TextPairBatch, i: int) -> Dict[str, Any]:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        log = dict(train_loss=loss)
        return dict(loss=loss, log=log)

    def validation_step(self, batch: TextPairBatch, i: int) -> Dict[str, Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = score_acc(logits, y)
        return dict(val_loss=loss, val_acc=acc)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = {}
        for d in outputs:
            for k, v in d.items():
                results.setdefault(k, []).append(v)
        log = {k: torch.stack(v).mean().item() for k, v in results.items()}
        print(log)
        return dict(val_loss=log["val_loss"], log=log)

    def test_step(self, batch: TextPairBatch, i: int) -> Dict[str, Tensor]:
        return self.validation_step(batch, i)

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.validation_epoch_end(outputs)

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

    def make_loader(self, ds: TextPairDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.config.bs,
            collate_fn=ds.collate_fn,
            shuffle=shuffle,
            num_workers=2,
        )

    def train_dataloader(self) -> DataLoader:
        return self.make_loader(self.make_dataset(SplitEnum.train), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.make_loader(self.make_dataset(SplitEnum.dev), shuffle=False)


def get_logger(save_dir: str, path_dotenv: str) -> CometLogger:
    load_dotenv(path_dotenv)
    assert os.getenv("COMET_KEY") is not None
    return CometLogger(
        api_key=os.getenv("COMET_KEY"),
        workspace=os.getenv("COMET_WORKSPACE"),  # Optional
        project_name=os.getenv("COMET_PROJECT"),  # Optional
        save_dir=save_dir,
        # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        # experiment_name=str(config),  # Optional
    )


class TextPairRetriever(Retriever):
    net: TextPairNet
    ds: TextGraphsQueryFactDataset

    def make_embeds(self, texts: List[str], bs: int = 64) -> Tensor:
        self.net.eval()
        with torch.no_grad():
            chunks = []
            for i in tqdm(range(0, len(texts), bs)):
                inputs = self.ds.make_tokens(texts[i : i + bs])
                chunks.append(self.net.embed_texts(inputs))

        embeds = torch.cat(chunks, dim=0)
        assert embeds.shape[0] == len(texts)
        return embeds

    def rank(self, queries: List[str], statements: List[str]) -> np.ndarray:
        embeds_a = self.make_embeds(queries)
        embeds_b = self.make_embeds(statements)
        distances = np.zeros(shape=(len(queries), len(statements)), dtype=np.float)

        self.net.eval()
        with torch.no_grad():
            for i in tqdm(range(len(queries))):
                a = embeds_a[i].unsqueeze(dim=0).repeat(len(statements), 1)
                x = self.net.fuse_embeds(a, embeds_b)
                x = self.net.run_head(x)
                assert x.shape == (len(statements), 2)
                x = nn.functional.softmax(x, dim=-1)
                dist: Tensor = x[:, 0]
                distances[i] = dist.numpy()

        ranking = np.argsort(distances, axis=-1)
        return ranking


def run_train(logger: CometLogger):
    config = TextPairConfig()
    system = TextPairSystem(**config.dict())
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.num_epochs,
        overfit_batches=config.overfit_pct,
        **get_device_kwargs(),
    )  # precision=16 is not faster on P100
    trainer.fit(system)


def run_eval(save_dir: str, data_split: SplitEnum):
    path = list(Path(save_dir).glob("**/*.ckpt"))[0]
    system = TextPairSystem.load_from_checkpoint(str(path))
    ds = system.make_dataset(data_split)
    retriever = TextPairRetriever(net=system.net, ds=ds)

    trainer = pl.Trainer(**get_device_kwargs())
    trainer.test(system, test_dataloaders=system.make_loader(ds, shuffle=False))

    # data = ds.data
    # preds = retriever.run(data)
    # Scorer().run(data.path_gold, preds)
    # ResultAnalyzer().run(data, preds)


def main(save_dir="/tmp/comet_logger", path_dotenv="../excluded/.env"):
    test_net()
    test_dataset()

    if not Path(save_dir).exists():
        logger = get_logger(save_dir, path_dotenv)
        run_train(logger)

    run_eval(save_dir, SplitEnum.dev)


if __name__ == "__main__":
    main()
