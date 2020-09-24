from abc import ABC
from typing import Any

import numpy as np
import torch
from allrank.models.losses import lambdaLoss
from torch import nn, Tensor

"""
Reference: https://github.com/almazan/deep-image-retrieval/blob/master/dirtorch/loss.py
"""


class APLoss(nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """

    def _forward_unimplemented(self, *args: Any) -> None:
        pass

    def __init__(self, nq=25, min_val=0, max_val=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min_val
        self.max = max_val
        gap = max_val - min_val
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2 * nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq - 1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(
            a * min_val + np.arange(nq, 0, -1)
        )  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(
            np.arange(2 - nq, 2, 1) - a * min_val
        )  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None, ret="1-mAP"):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, : self.nq], q[:, self.nq :]).clamp(
            min=0, max=np.inf
        )  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(
            dim=-1
        )  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == "1-mAP":
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == "AP":
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {"loss_ap": float(loss)}


def test_loss(loss_fn: nn.Module):
    num_queries, num_facts = 32, 64
    threshold = 0.5

    x = torch.randn(num_queries, num_facts)
    y = torch.gt(torch.clone(x), threshold).float()
    print(dict(x=x.dtype, y=y.dtype))
    loss = loss_fn(x, y)
    print(dict(loss=loss.item()))


class TAPLoss(APLoss):
    """ Differentiable tie-aware AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """

    def __init__(self, nq=25, min_val=0, max_val=1, simplified=False):
        APLoss.__init__(self, nq=nq, min_val=min_val, max_val=max_val)
        self.simplified = simplified

    def forward(self, x, label, qw=None, ret="1-mAP"):
        """N: number of images;
           M: size of the descs;
           Q: number of bins (nq);
        """
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        label = label.float()
        Np = label.sum(dim=-1, keepdim=True)

        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, : self.nq], q[:, self.nq :]).clamp(
            min=0, max=np.inf
        )  # N x Q x M

        c = q.sum(dim=-1)  # number of samples  N x Q = nbs on APLoss
        cp = (q * label.view(N, 1, M)).sum(
            dim=-1
        )  # N x Q number of correct samples = rec on APLoss
        C = c.cumsum(dim=-1)
        Cp = cp.cumsum(dim=-1)

        zeros = torch.zeros(N, 1).to(x.device)
        C_1d = torch.cat((zeros, C[:, :-1]), dim=-1)
        Cp_1d = torch.cat((zeros, Cp[:, :-1]), dim=-1)

        if self.simplified:
            aps = cp * (Cp_1d + Cp + 1) / (C_1d + C + 1) / Np
        else:
            eps = 1e-8
            ratio = (cp - 1).clamp(min=0) / ((c - 1).clamp(min=0) + eps)
            aps = (
                cp
                * (
                    c * ratio
                    + (Cp_1d + 1 - ratio * (C_1d + 1)) * torch.log((C + 1) / (C_1d + 1))
                )
                / (c + eps)
                / Np
            )
        aps = aps.sum(dim=-1)

        assert aps.numel() == N

        if ret == "1-mAP":
            if qw is not None:
                aps *= qw  # query weights
            return 1 - aps.mean()
        elif ret == "AP":
            assert qw is None
            return aps
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {"loss_tap" + ("s" if self.simplified else ""): float(loss)}


class LambdaLoss(nn.Module, ABC):
    # Reference: https://github.com/allegro/allRank/blob/master/tests/losses/test_lambdaloss.py
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return lambdaLoss(y_pred, y_true, **self.kwargs, reduction="mean")


def main():
    test_loss(loss_fn=APLoss())
    test_loss(loss_fn=TAPLoss())
    test_loss(loss_fn=LambdaLoss())


if __name__ == "__main__":
    main()
