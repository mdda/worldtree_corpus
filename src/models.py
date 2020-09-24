from abc import ABC
from copy import deepcopy
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import GCNConv  # noqa
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    AutoConfig,
)
from transformers.modeling_distilbert import FFN
from transformers.modeling_outputs import BaseModelOutput

"""
After each transformer layer, output is [bs, seq_len, dim]
Normally sequences across bs are independent
But we are feeding in each batch ordered documents for a single question for ranking
Normally pooled output is [:, 0, :] of last layer only
We want a LSTM-adapter after each layer to carry relations across documents
But the intermediate layers don't have a special conditioned [CLS] position like last layer!
Mean pool? Or max pool? Try mean pool first
But how to feed back into transformer? Maybe just standardize [CLS] position like last layer
Concern of too many parameters of LSTM especially with 768-dim
Use down-and-up-projection like Adapters
"""


class GcnBlock(nn.Module, ABC):
    def __init__(self, input_size: int, hidden_size: int, p_dropout: float):
        super().__init__()
        self.gcn = GCNConv(input_size, hidden_size, add_self_loops=False)
        self.final = nn.Sequential(nn.ReLU(), nn.Dropout(p=p_dropout))

    def forward(self, *args) -> Tensor:
        x = self.gcn(*args)
        x = self.final(x)
        return x


class Interpolator(nn.Module, ABC):
    """
    interpolate(a, b) = w * a + (1-w) * b
    sigmoid(8.0) ~= 0.9996 which "selects" mostly a by default
    """

    def __init__(self, init_value: float = 8.0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.full(size=(1,), fill_value=init_value, dtype=torch.float)
        )
        print(f"Interpolate init={init_value}, sigmoid={self.test_sigmoid(init_value)}")

    @staticmethod
    def test_sigmoid(x: float) -> float:
        e = 2.7182  # Approximate
        power = -1 * x
        x = 1 / (1 + e ** power)
        return x

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        assert a.shape == b.shape
        w = torch.sigmoid(self.weight).expand_as(a)
        x = w * a + torch.sub(1.0, w) * b
        return x


def test_interpolator():
    a = torch.ones(3, 3)
    b = torch.zeros_like(a)
    net = Interpolator()
    print(net(a, b))


class SequenceAdapter(nn.Module, ABC):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class RnnAdapter(SequenceAdapter, ABC):
    def __init__(self, hidden_size: int, project_size: int, p_dropout=0.0):
        super().__init__()
        self.down = nn.Linear(hidden_size, project_size)
        self.up = nn.Linear(project_size, hidden_size)
        self.rnn = nn.LSTM(
            input_size=project_size,
            hidden_size=project_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.interpolator = Interpolator()
        self.dropout = nn.Dropout(p=p_dropout)

    def forward_rnn(self, x: Tensor) -> Tensor:
        num, dim = x.shape
        x = torch.unsqueeze(x, dim=0)
        x, states = self.rnn(x)
        assert tuple(x.shape) == (1, num, dim)
        x = torch.squeeze(x, dim=0)
        return x

    def forward_pooled(self, inputs: Tensor) -> Tensor:
        x = inputs
        x = F.gelu(self.down(x))
        x = self.forward_rnn(x)
        x = F.gelu(self.up(x))
        x = self.dropout(x)
        x = self.interpolator(inputs, x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        num, length, dim = x.shape
        pooled = self.forward_pooled(x[:, 0, :])
        pooled = pooled.unsqueeze(dim=1)
        x = torch.cat([pooled, x[:, 1:, :]], dim=1)
        assert tuple(x.shape) == (num, length, dim)
        return x


def test_adapter():
    num, length, dim = 32, 100, 768
    hidden_size, project_size = dim, dim // 2

    net = RnnAdapter(hidden_size, project_size)
    x = torch.zeros(num, length, dim, dtype=torch.float, requires_grad=True)
    outputs = net(x)
    print(net(x).shape)
    torch.mean(outputs).backward()
    print(x.grad)


class AdaptedFFN(nn.Module, ABC):
    def __init__(self, ffn: nn.Module, adapter: SequenceAdapter, is_verbose=False):
        super().__init__()
        self.ffn = ffn
        self.adapter = adapter
        self.is_verbose = is_verbose

    def forward(self, x: Tensor) -> Tensor:
        num, length, dim = x.shape
        x = self.ffn(x)
        x = self.adapter(x)
        assert tuple(x.shape) == (num, length, dim)
        if self.is_verbose:
            print(dict(forward="AdaptedFFN"))
        return x


def test_ffn():
    num, length, dim = 32, 100, 768
    config = AutoConfig.from_pretrained("ishan/distilbert-base-uncased-mnli")
    hidden_size = config.dim
    project_size = hidden_size // 2
    ffn = FFN(config)

    adapter = RnnAdapter(hidden_size=hidden_size, project_size=project_size)
    net = AdaptedFFN(ffn=ffn, adapter=adapter, is_verbose=True)

    x = torch.zeros(num, length, dim, dtype=torch.float)
    print(net(x).shape)


class AdaptedTransformer(nn.Module, ABC):
    def __init__(
        self, transformer: Union[PreTrainedModel, nn.Module], adapter: SequenceAdapter
    ):
        super().__init__()
        self.transformer = transformer
        for i, block in enumerate(self.transformer.transformer.layer):
            transformer.transformer.layer[i] = self.patch_block(block, adapter)

    @staticmethod
    def patch_block(block: nn.Module, adapter: SequenceAdapter) -> nn.Module:
        adapter = deepcopy(adapter)
        block.ffn = AdaptedFFN(ffn=block.ffn, adapter=adapter)
        return block

    def forward(self, **kwargs):
        return self.transformer(**kwargs)


def print_transformer_outputs(x: BaseModelOutput) -> List[dict]:
    return [dict(sequence_index=i, x=x.last_hidden_state[:, i, :]) for i in [0, 1]]


def test_transformer():
    model_name = "ishan/distilbert-base-uncased-mnli"
    texts = ["this is a sentence", "this is another"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    net = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    outputs = net(**inputs, return_dict=True)
    print(dict(outputs=print_transformer_outputs(outputs)))

    hidden_size = net.config.dim
    project_size = hidden_size // 2
    adapter = RnnAdapter(hidden_size=hidden_size, project_size=project_size)
    net_new = AdaptedTransformer(transformer=net, adapter=adapter)
    outputs_new = net_new(**inputs, return_dict=True)
    print(dict(outputs_new=print_transformer_outputs(outputs_new)))


def main():
    test_interpolator()
    test_adapter()
    test_ffn()
    test_transformer()


if __name__ == "__main__":
    main()
