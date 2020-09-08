from typing import Tuple, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import BaseModelOutput, TokenClassifierOutput

from extra_data import (
    MSMarcoHierarchicalSentenceDataset,
    MSMarcoExample,
    SplitEnum,
    KEY_LABEL,
)

assert MSMarcoExample is not None  # Needed for pickle.load in MSMarcoDataset


class HierarchicalSentenceClassifier(PreTrainedModel):
    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self, model_tokens: PreTrainedModel, model_sents: PreTrainedModel):
        super().__init__(model_sents.config)
        self.model_sents = model_sents
        self.model_tokens = model_tokens
        self.num_labels = model_sents.config.num_labels

    def forward_token_level(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        bs, num_seq, seq_len = input_ids.shape
        assert bs == 1
        outputs: BaseModelOutput = self.model_tokens(
            input_ids=input_ids.squeeze(dim=0),
            attention_mask=attention_mask.squeeze(dim=0),
            return_dict=True,
        )
        _, _, dim = outputs.last_hidden_state.shape
        pooled: Tensor = outputs.last_hidden_state[:, 0, :]
        assert tuple(pooled.shape) == (num_seq, dim)
        return pooled

    def forward_sent_level(
        self, sent_embeds: Tensor, labels: Tensor = None
    ) -> TokenClassifierOutput:
        num_seq, dim = sent_embeds.shape
        sent_embeds = torch.unsqueeze(sent_embeds, dim=0)
        if labels is not None:
            assert tuple(labels.shape) == (1, num_seq)
        return self.model_sents(
            inputs_embeds=sent_embeds, labels=labels, return_dict=True
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor = None):
        sent_embeds = self.forward_token_level(input_ids, attention_mask)
        return self.forward_sent_level(sent_embeds, labels)


def get_model(
    name: str, num_labels: int
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(name)
    model_tokens = AutoModel.from_pretrained(name)
    model_sents = AutoModelForTokenClassification.from_pretrained(
        name, num_labels=num_labels
    )
    model = HierarchicalSentenceClassifier(model_tokens, model_sents)
    print(list(map(type, [tokenizer, model_tokens, model_sents, model_sents.config])))
    return tokenizer, model


def test_model():
    multiplier = 9  # max_sequences(model=DistilBERT, device="cuda")=900
    texts = [
        "We are very happy to show you the transformers library",
        "The quick fox jumped over the lazy brown dog",
        "This huggingface library is not bad",
    ] * multiplier
    labels: Tensor = Tensor([0, 1, 0] * multiplier).long()
    tokenizer, model = get_model(name="distilbert-base-uncased", num_labels=2)
    inputs: BatchEncoding = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    )
    kwargs = {k: v.unsqueeze(dim=0) for k, v in inputs.items()}
    model(**kwargs)
    kwargs[KEY_LABEL] = labels.unsqueeze(dim=0)

    device = torch.device("cpu")
    model = model.to(device)
    kwargs = {k: v.to(device) for k, v in kwargs.items()}
    outputs: TokenClassifierOutput = model(**kwargs)
    info = dict(
        inputs={k: v.shape for k, v in kwargs.items()},
        outputs={k: v.shape for k, v in outputs.items()},
    )
    print(info)


class DummyDataset(Dataset):
    def __init__(self):
        self.num_seq_per_example = 128
        self.max_seq_len = 128

    def __getitem__(self, i: int) -> BatchEncoding:
        input_ids = torch.ones(
            self.num_seq_per_example, self.max_seq_len, dtype=torch.long
        )
        data = dict(
            input_ids=input_ids,
            attention_mask=torch.clone(input_ids),
            labels=torch.ones(self.num_seq_per_example, dtype=torch.long),
        )
        return BatchEncoding(data=data)

    def __len__(self) -> int:
        return 10000


def test_dummy_dataset(model: PreTrainedModel, train_args):
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
    )
    trainer.train()


def run_train(num_labels=2, bs=1):
    tokenizer, model = get_model(name="distilbert-base-uncased", num_labels=num_labels)
    train_args = TrainingArguments(
        output_dir=f"/tmp/{type(model).__name__}",
        do_train=True,
        do_eval=True,
        evaluate_during_training=True,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        save_total_limit=10,
    )
    test_dummy_dataset(model, train_args)

    ds_train = MSMarcoHierarchicalSentenceDataset(SplitEnum.train, tokenizer)
    ds_dev = MSMarcoHierarchicalSentenceDataset(SplitEnum.dev, tokenizer)
    trainer = Trainer(
        model=model, args=train_args, train_dataset=ds_train, eval_dataset=ds_dev
    )
    trainer.train()


if __name__ == "__main__":
    test_model()
    run_train()
