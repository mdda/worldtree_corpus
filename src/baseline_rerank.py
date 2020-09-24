import os
import random
from collections import Counter
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Set, Callable, Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from comet_ml import BaseExperiment
from dotenv import load_dotenv
from pydantic import BaseModel
from pytorch_lightning.loggers import CometLogger
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from transformers import (
    BatchEncoding,
    AutoTokenizer,
    PreTrainedModel,
    AutoModel,
    AutoModelForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput, TokenClassifierOutput

from baseline_retrieval import (
    Data,
    Retriever,
    Prediction,
    ResultAnalyzer,
    Scorer,
    PredictManager,
    SpacyProcessor,
    TextProcessor,
)
from dataset import QuestionAnswer, ExplanationUsed, TxtAndKeywords
from extra_data import SplitEnum, analyze_lengths
from losses import APLoss, TAPLoss, LambdaLoss
from vectorizers import BM25Vectorizer

pl.seed_everything(42)


class NetEnum(str, Enum):
    transformer = "transformer"
    rnn = "rnn"
    dense = "dense"
    gcn = "gcn"


class Config(BaseModel):
    bs: int = 1
    top_n: int = 128
    model_name: str = "ishan/distilbert-base-uncased-mnli"
    num_labels: int = 1
    num_bonus_contexts: int = 0
    add_missing_gold: bool = True

    net_features_type: NetEnum = NetEnum.transformer

    net_ranker_type: NetEnum = NetEnum.rnn
    net_ranker_num_layers: int = 2
    net_ranker_input_size: int = 768
    net_ranker_hidden_size: int = 128

    p_dropout: float = 0.1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    num_epochs: int = 3
    overfit_pct: float = 0.0

    data_name: str = "textgraphs"
    loss_name: str = "bce"
    input_pattern: str = "../predictions/predict.FOLD.baseline-retrieval.txt"
    output_pattern: str = "../predictions/predict.FOLD.baseline-rerank.txt"


class Example(BaseModel):
    query: str
    docs: List[str]
    labels: List[int]
    gold: Set[str]


class NetInputs(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    encoding: BatchEncoding
    graph: GraphData = GraphData()

    def to(self, device: str):
        self.encoding = self.encoding.to(device)
        self.graph = self.graph.to(device)
        return self


class RankerNet(nn.Module):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def forward(self, inputs: NetInputs) -> Tensor:
        raise NotImplementedError


class RnnRankerNet(RankerNet):
    def __init__(self, config: Config):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=config.net_ranker_input_size,
            hidden_size=config.net_ranker_hidden_size,
            num_layers=config.net_ranker_num_layers,
            dropout=config.p_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(config.net_ranker_hidden_size * 2, 1), nn.Sigmoid()
        )

    def forward(self, inputs: NetInputs) -> Tensor:
        x = inputs.graph.x
        num, dim = x.shape
        x = x.unsqueeze(dim=0)
        x, states = self.rnn(x)
        x = self.linear(x)
        x = torch.squeeze(x, dim=-1)
        assert tuple(x.shape) == (1, num)
        return x


class GcnBlock(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, input_size: int, hidden_size: int, p_dropout: float):
        super().__init__()
        self.gcn = GCNConv(input_size, hidden_size, add_self_loops=False)
        self.final = nn.Sequential(nn.ReLU(), nn.Dropout(p=p_dropout))

    def forward(self, *args) -> Tensor:
        x = self.gcn(*args)
        x = self.final(x)
        return x


class GcnRankerNet(RankerNet):
    def __init__(self, config: Config):
        super().__init__()
        input_size = config.net_ranker_input_size
        hidden_size = config.net_ranker_hidden_size
        self.config = config

        self.conv1 = GcnBlock(input_size, hidden_size, config.p_dropout)
        self.conv2 = GcnBlock(hidden_size, hidden_size, config.p_dropout)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(config.net_ranker_hidden_size, 1), nn.Sigmoid()
        )

    def forward_rnn(self, x: Tensor) -> Tensor:
        num, dim = x.shape
        x = torch.unsqueeze(x, dim=0)
        x, states = self.rnn(x)
        x = torch.squeeze(x, dim=0)
        assert tuple(x.shape) == (num, dim)
        return x

    def forward(self, inputs: NetInputs):
        x = inputs.graph.x
        num, dim = x.shape
        indices = inputs.graph.edge_index
        weights = inputs.graph.edge_attr

        x = self.conv1(x, indices, weights)
        x = self.conv2(x, indices, weights)
        x = self.forward_rnn(x)
        x = self.linear(x)

        assert tuple(x.shape) == (num, 1)
        x = torch.transpose(x, 0, 1)
        return x


class TransformerRankerNet(RankerNet):
    def __init__(self, config: Config):
        super().__init__()
        self.transformer: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            config.model_name, num_labels=1
        )

    def forward(self, inputs: NetInputs) -> Tensor:
        x = inputs.graph.x
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
    def _forward_unimplemented(self, *args: Any) -> None:
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

    def forward(self, inputs: NetInputs) -> Tensor:
        inputs.graph.x = self.net_features(inputs.encoding)
        x = self.net_ranker(inputs)
        return x


def make_net(config: Config) -> nn.Module:
    class_ranker = {
        NetEnum.rnn: RnnRankerNet,
        NetEnum.transformer: TransformerRankerNet,
        NetEnum.gcn: GcnRankerNet,
    }[config.net_ranker_type]
    net_ranker = class_ranker(config)

    class_features = {NetEnum.transformer: TransformerFeatureNet}[
        config.net_features_type
    ]
    net_features = class_features(config)
    net = HierarchicalRankerNet(net_features, net_ranker)
    return net


def test_net(
    inputs: NetInputs = None,
    labels: Tensor = None,
    config=Config(),
    num_seq=128,
    seq_len=40,
):
    if inputs is None:
        x = torch.ones(num_seq, seq_len, dtype=torch.long)
        graph_maker = GraphMaker()
        texts = list("this is dummy text which will be split up by characters")
        inputs = NetInputs(
            encoding=BatchEncoding(data=dict(input_ids=x, attention_mask=x)),
            graph=graph_maker.run(texts),
        )

    if labels is None:
        labels = torch.lt(torch.randn(config.bs, num_seq), 0.05).float()

    net = make_net(config)
    outputs = net(inputs)
    assert tuple(outputs.shape) == (config.bs, num_seq)
    assert outputs.shape == labels.shape
    loss_fn = System.make_loss_fn(config)
    loss = loss_fn(outputs, labels)
    print(dict(loss=loss))


class GraphMaker(BaseModel):
    vectorizer: TfidfVectorizer = BM25Vectorizer()
    preproc: TextProcessor = SpacyProcessor(remove_stopwords=True)

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def csr_to_edges(x: csr_matrix) -> Tuple[Tensor, Tensor]:
        x_coo: coo_matrix = x.tocoo()
        rows = torch.from_numpy(x_coo.row).unsqueeze(dim=0)
        cols = torch.from_numpy(x_coo.col).unsqueeze(dim=0)
        indices = torch.cat([rows, cols], dim=0)
        indices = indices.long()

        values = torch.from_numpy(x_coo.data).float()
        values = values.unsqueeze(dim=-1)
        num_edges, num_features = values.shape
        assert tuple(indices.shape) == (2, num_edges)
        return indices, values

    def run(self, texts: List[str]) -> Data:
        texts = [self.preproc.run(TxtAndKeywords(raw_txt=t)) for t in texts]
        vectors = self.vectorizer.fit_transform(texts)
        scores: csr_matrix = cosine_similarity(vectors, dense_output=False)
        indices, values = self.csr_to_edges(scores)
        return GraphData(edge_index=indices, edge_attr=values)


class RerankDataset(Dataset):
    """
    Use ranker to get preds
    For each question
        Get top_n preds + gold explanations
        Fix bs=1 to avoid complication
        Target (tensor) shape is (bs, top_n) in {0, 1}
        Input (BatchEncoding) shape is (bs, top_n, seq_len)
    """

    def __init__(self, data: Data, config: Config, is_test=False):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.data = data
        self.is_train = self.data.data_split == SplitEnum.train
        self.manager = PredictManager(file_pattern=config.input_pattern)
        self.top_n = config.top_n
        self.uid_to_text = {s.uid: s.raw_txt for s in self.data.statements}
        self.examples: List[Example] = [] if is_test else self.load()
        self.graph_maker = GraphMaker()

    def qn_to_text(self, q: QuestionAnswer, p: Prediction):
        texts = self.pred_to_texts(p, top_n=self.config.num_bonus_contexts)
        texts.append(q.question.raw_txt)
        texts.append(q.answers[0].raw_txt)
        return " ".join(texts)

    @property
    def preds_and_qns(self) -> Tuple[List[Prediction], List[QuestionAnswer]]:
        preds = self.manager.read_pickle(self.data.data_split)
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
        # Cancel last: Don't add missing gold, just take top_n first to avoid complications
        e: ExplanationUsed
        return p.uids[: self.top_n], set([e.uid for e in q.explanation_gold])

    def pred_to_texts(self, p: Prediction, top_n: int) -> List[str]:
        return [self.uid_to_text[u] for u in p.uids[:top_n]]

    def qn_to_explains(self, q: QuestionAnswer) -> List[str]:
        explains = []
        for e in q.explanation_gold:
            text = self.uid_to_text.get(e.uid)
            if text is None:
                print(dict(qn_explain_not_found=e.uid))
            else:
                explains.append(text)
        return explains

    def load(self) -> List[Example]:
        preds, qns = self.preds_and_qns
        examples = []
        for p, q in zip(preds, qns):
            docs_gold = set(self.qn_to_explains(q))
            docs = self.pred_to_texts(p, self.top_n)
            labels = [int(d in docs_gold) for d in docs]
            if sum(labels) == 0:
                if not (self.is_train and self.config.add_missing_gold):
                    print(dict(no_hits_in_top_n=q.question_id))
                    continue
            examples.append(
                Example(
                    query=self.qn_to_text(q, p),
                    docs=docs,
                    labels=labels,
                    gold=docs_gold,
                )
            )
        return examples

    def make_tokens(self, texts_a: List[str], texts_b: List[str]) -> BatchEncoding:
        return self.tokenizer(texts_a, texts_b, padding=True, return_tensors="pt")

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def insert_gold_missing(e: Example) -> Example:
        e = deepcopy(e)
        indices_neg = [i for i, x in enumerate(e.labels) if x == 0]
        docs = set(e.docs)
        gold_missing = [text for text in e.gold if text not in docs]
        assert len(indices_neg) >= len(gold_missing)
        random.shuffle(indices_neg)

        for i, text in enumerate(gold_missing):
            j = indices_neg[i]
            assert e.labels[j] == 0
            e.docs[j] = text
            e.labels[j] = 1

        return e

    @property
    def collate_fn(self) -> Callable:
        def fn(examples: List[Example]) -> Tuple[NetInputs, Tensor]:
            assert len(examples) == 1
            e = examples[0]
            if self.is_train and self.config.add_missing_gold:
                e = self.insert_gold_missing(e)

            docs = e.docs
            queries = [e.query] * len(docs)
            y = Tensor([e.labels])
            inputs = NetInputs(
                encoding=self.make_tokens(queries, docs),
                graph=self.graph_maker.run(docs),
            )
            return inputs, y

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


def test_dataset(data_split=SplitEnum.dev, config=Config()):
    data = Data(data_split=data_split)
    data.load()
    dataset = RerankDataset(data, config)

    # Num workers is not too important: default(0) -> 49 it/s, 4 -> 79 it/s but bottleneck is likely model
    loader = DataLoader(
        dataset,
        batch_size=config.bs,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=os.cpu_count(),
    )
    limit = 100
    for i, _ in tqdm(enumerate(loader), total=limit):
        if i == limit:
            break

    inputs: NetInputs
    inputs, labels = next(iter(loader))
    num_seq, seq_len = inputs.encoding.input_ids.shape
    assert tuple(labels.shape) == (config.bs, num_seq)
    test_net(inputs=inputs, labels=labels, config=config)
    dataset.analyze()


class RerankRetriever(Retriever):
    """
    Run base retriever on data to get preds
    For each query, pred
        Get query-doc pair texts for top_n
        Tokenize
        Feed to model to get new scores for top_n
        Re-rank top_n by scores
    Re-run scorer/analyzer
    """

    manager: PredictManager
    net: HierarchicalRankerNet
    dataset: RerankDataset
    top_n: int
    device: str = "cuda"

    def run_qn(
        self, q: QuestionAnswer, p: Prediction, net: HierarchicalRankerNet
    ) -> Tensor:
        query = self.dataset.qn_to_text(q, p)
        docs = self.dataset.pred_to_texts(p, top_n=self.top_n)
        ex = Example(query=query, docs=docs, labels=[0] * len(docs), gold=set())
        x: BatchEncoding
        x, y = self.dataset.collate_fn([ex])
        x = x.to(self.device)
        with torch.no_grad():
            scores = net(x)

        assert tuple(scores.shape) == (1, len(docs))
        scores = torch.squeeze(scores, dim=0)
        return scores

    def run(self, data: Data) -> List[Prediction]:
        preds = self.manager.read_pickle(data.data_split)
        assert len(preds) == len(data.questions)
        scores = torch.zeros(len(preds), self.top_n, device=self.device)
        net = self.net.eval().to(self.device)

        for i, q in tqdm(enumerate(data.questions)):
            scores[i] = self.run_qn(q, preds[i], net)
        distances = np.multiply(scores.cpu().numpy(), -1)
        reranking = np.argsort(distances, axis=-1)

        for i, p in enumerate(preds):
            uids_rerank = [p.uids[: self.top_n][j] for j in reranking[i]]
            assert len(uids_rerank) == self.top_n
            p.uids[: self.top_n] = uids_rerank
        return preds


class System(pl.LightningModule):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = kwargs  # For logging
        self.config = Config(**kwargs)
        self.net = make_net(self.config)
        self.loss_fn = self.make_loss_fn(self.config)
        self.ds_train = self.make_dataset(SplitEnum.train)
        self.ds_dev = self.make_dataset(SplitEnum.dev)

    @staticmethod
    def make_loss_fn(config: Config) -> nn.Module:
        mapping = dict(
            lambdaloss=LambdaLoss(),
            bce=nn.BCELoss(),
            mse=nn.MSELoss(),
            crossentropy=nn.CrossEntropyLoss(),
            map=APLoss(),
            tap=TAPLoss(),
        )
        return mapping[config.loss_name]

    def make_dataset(self, data_split: SplitEnum) -> RerankDataset:
        data = Data(data_split=data_split)
        data.load()
        mapping = dict(
            textgraphs=RerankDataset(
                data, self.config, is_test=(data_split == SplitEnum.test)
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

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = {}
        for d in outputs:
            for k, v in d.items():
                results.setdefault(k, []).append(v)
        log = {k: torch.stack(v).mean().item() for k, v in results.items()}
        # log.update(val_map=self.run_eval(SplitEnum.dev))
        print(log)
        return dict(val_loss=log["val_loss"], log=log)

    def test_step(
        self, batch: Tuple[BatchEncoding, Tensor], i: int
    ) -> Dict[str, Tensor]:
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
        # experiment_name=str(config),  # Optional
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


def run_eval(save_dir: str, data_split=SplitEnum.dev):
    path = list(Path(save_dir).glob("**/*.ckpt"))[0]
    print(path)
    system = System.load_from_checkpoint(str(path))
    manager_in = PredictManager(file_pattern=system.config.input_pattern)
    manager_out = PredictManager(file_pattern=system.config.output_pattern)

    ds = system.make_dataset(data_split)
    retriever = RerankRetriever(
        net=system.net, dataset=ds, top_n=system.config.top_n, manager=manager_in
    )

    if data_split != SplitEnum.test:
        trainer = pl.Trainer(gpus=1)
        trainer.test(system, test_dataloaders=system.make_loader(ds, shuffle=False))

    data = ds.data
    preds = retriever.run(data)
    manager_out.write(preds, data_split)
    if data_split != SplitEnum.test:
        Scorer().run(data.path_gold, manager_out.make_path(data_split))
        ResultAnalyzer().run(data, preds)


def main(save_dir="/tmp/comet_logger", path_dotenv="../excluded/.env"):
    if not Path(save_dir).exists():
        logger = get_logger(save_dir, path_dotenv)
        run_train(logger)
    run_eval(save_dir, data_split=SplitEnum.dev)
    run_eval(save_dir, data_split=SplitEnum.test)


if __name__ == "__main__":
    test_net()
    test_dataset()
    main()
