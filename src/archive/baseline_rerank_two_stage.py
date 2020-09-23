import os
from pathlib import Path
from typing import Tuple, Any, Dict, List

import pytorch_lightning as pl
import torch
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW, BatchEncoding

from baseline_rerank import (
    System,
    RerankRetriever,
    NetEnum,
    FeatureNet,
    RankerNet,
    RerankDataset,
    HierarchicalRankerNet,
)
from baseline_retrieval import ResultAnalyzer, Scorer, PredictManager, Prediction, Data
from extra_data import SplitEnum, hash_text

print(Prediction, "Importing this is required for pickle loading")


class Config(BaseModel):
    bs: int = 1
    model_name: str = "ishan/distilbert-base-uncased-mnli"
    num_labels: int = 1
    num_bonus_contexts: int = 0

    net_features_type: NetEnum = NetEnum.transformer

    net_ranker_type: NetEnum = NetEnum.rnn
    net_ranker_num_layers: int = 2
    net_ranker_input_size: int = 768
    net_ranker_hidden_size: int = 128

    p_dropout: float = 0.1
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    overfit_pct: float = 0.0

    num_epochs: int = 1
    num_epochs_b: int = 10
    top_n: int = 128
    top_n_b: int = 512
    learning_rate: float = 5e-5
    learning_rate_b: float = 5e-4
    save_path: str = "system.ckpt"
    save_path_b: str = "system_b.ckpt"

    data_name: str = "textgraphs"
    loss_name: str = "map"
    input_pattern: str = "../predictions/predict.FOLD.baseline-retrieval.txt"
    output_pattern: str = "../predictions/predict.FOLD.baseline-rerank-two-stage.txt"


def run_train(config: Config):
    system = System(**config.dict())
    trainer = pl.Trainer(
        max_epochs=config.num_epochs, overfit_batches=config.overfit_pct, gpus=1,
    )  # precision=16 is not faster on P100
    trainer.fit(system)
    trainer.save_checkpoint(config.save_path)


def run_eval(path: str, data_split=SplitEnum.dev):
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


class RerankBDataset(Dataset):
    def __init__(
        self,
        data: Data,
        config: Config,
        net: FeatureNet,
        root="/tmp/rerank_b_dataset",
        device="cuda",
    ):
        self.net = net.eval()
        self.config = config

        data.load()
        ds = RerankDataset(data, config, is_test=True)
        ds.top_n = config.top_n_b
        ds.examples = ds.load()

        self.root = root + str(len(ds.examples))
        self.paths = self.load(ds, device)

    def save_data(self, x: Tensor, y: Tensor):
        data = (x.cpu(), y.cpu())
        hash_id = hash_text(str(data))

        path = Path(self.root) / f"{hash_id}.pt"
        if not path.parent.exists():
            path.parent.mkdir()
        if not path.exists():
            torch.save(data, str(path))

    def load(self, ds: RerankDataset, device: str) -> List[Path]:
        if not Path(self.root).exists():
            loader = DataLoader(
                ds,
                batch_size=self.config.bs,
                collate_fn=ds.collate_fn,
                shuffle=False,
                num_workers=os.cpu_count(),
            )
            net = self.net.eval().to(device)
            with torch.no_grad():
                for x, y in tqdm(loader, desc="make_dataset"):
                    x = x.to(device)
                    x = net(x)
                    self.save_data(x, y)

        return sorted(Path(self.root).iterdir())

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        return torch.load(str(self.paths[i]))

    def __len__(self) -> int:
        return len(self.paths)


class SystemB(pl.LightningModule):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self, **kwargs):
        super().__init__()
        self.hparams = kwargs  # For logging
        self.config = Config(**kwargs)

        self.system_base = System.load_from_checkpoint(self.config.save_path)
        self.net_features: FeatureNet = self.system_base.net.net_features
        self.net: RankerNet = self.system_base.net.net_ranker

        self.loss_fn = self.system_base.make_loss_fn()
        self.ds_dev = self.make_dataset(SplitEnum.dev)
        self.ds_train = self.make_dataset(SplitEnum.train)

    def forward(self, x: Tensor):
        return self.net(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], i: int) -> Dict[str, Any]:
        x, y = batch
        x, y = x.squeeze(dim=0), y.squeeze(dim=0)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        log = dict(train_loss=loss)
        return dict(loss=loss, log=log)

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], i: int
    ) -> Dict[str, Tensor]:
        x, y = batch
        x, y = x.squeeze(dim=0), y.squeeze(dim=0)
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

    def test_step(self, batch: Tuple[Tensor, Tensor], i: int) -> Dict[str, Tensor]:
        return self.validation_step(batch, i)

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.validation_epoch_end(outputs)

    def make_dataset(self, data_split: SplitEnum) -> RerankBDataset:
        data = Data(data_split=data_split)
        return RerankBDataset(data, self.config, self.net_features)

    def configure_optimizers(self):
        return AdamW(self.net.parameters(), self.config.learning_rate_b)

    def make_loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.config.bs, shuffle=shuffle, num_workers=os.cpu_count(),
        )

    def train_dataloader(self) -> DataLoader:
        return self.make_loader(self.ds_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.make_loader(self.ds_dev, shuffle=False)


def run_train_b(config: Config):
    system = SystemB(**config.dict())
    trainer = pl.Trainer(
        max_epochs=config.num_epochs_b, overfit_batches=config.overfit_pct, gpus=1,
    )  # precision=16 is not faster on P100
    trainer.fit(system)
    trainer.save_checkpoint(config.save_path_b)


def run_eval_b(save_dir: str, data_split=SplitEnum.dev):
    path = sorted(Path(save_dir).glob("**/*.ckpt"))[-1]
    system = SystemB.load_from_checkpoint(str(path))
    manager_in = PredictManager(file_pattern=system.config.input_pattern)
    manager_out = PredictManager(file_pattern=system.config.output_pattern)

    ds = system.system_base.make_dataset(data_split)
    net = HierarchicalRankerNet(net_features=system.net_features, net_ranker=system.net)
    retriever = RerankRetriever(
        net=net, dataset=ds, top_n=system.config.top_n_b, manager=manager_in
    )

    # if data_split != SplitEnum.test:
    #     trainer = pl.Trainer(gpus=1)
    #     trainer.test(system, test_dataloaders=system.make_loader(ds, shuffle=False))

    data = ds.data
    preds = retriever.run(data)
    manager_out.write(preds, data_split)
    if data_split != SplitEnum.test:
        Scorer().run(data.path_gold, manager_out.make_path(data_split))
        ResultAnalyzer().run(data, preds)


def main():
    # config = Config()
    #
    # if not Path(config.save_path).exists():
    #     run_train(config)
    #     run_eval(config.save_path, data_split=SplitEnum.dev)
    #
    # if not Path(config.save_path_b).exists():
    #     run_train_b(config)

    run_eval_b("lightning_logs/version_0/checkpoints/", data_split=SplitEnum.test)


if __name__ == "__main__":
    main()
