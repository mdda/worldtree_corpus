from typing import Optional, List

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)

from baseline_retrieval import Retriever, Data, PredictManager, Scorer, ResultAnalyzer
from extra_data import SplitEnum


class Encoder(BaseModel):
    name: str
    net: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    device: str = "cuda"
    bs: int = 128

    class Config:
        arbitrary_types_allowed = True

    def load(self):
        if "question" in self.name:
            self.net = DPRQuestionEncoder.from_pretrained(self.name)
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.name)
        elif "ctx" in self.name:
            self.net = DPRContextEncoder.from_pretrained(self.name)
            self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.name)
        else:
            raise ValueError(str(dict(invalid_name=self.name)))

    def run(self, texts: List[str]) -> Tensor:
        if self.net is None or self.tokenizer is None:
            self.load()
        net = self.net.eval().to(self.device)

        outputs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.bs)):
                batch = texts[i : i + self.bs]
                inputs = self.tokenizer(batch, padding=True, return_tensors="pt")
                inputs = inputs.to(self.device)
                outputs.append(net(**inputs, return_dict=True).pooler_output)

        return torch.cat(outputs, dim=0)


class DprRetriever(Retriever):
    encoder_q: Encoder
    encoder_d: Encoder

    def rank(self, queries: List[str], statements: List[str]) -> np.ndarray:
        vecs_q = self.encoder_q.run(queries)
        vecs_d = self.encoder_d.run(statements)
        scores = torch.matmul(vecs_q, torch.transpose(vecs_d, 0, 1))
        scores = scores.cpu()
        return np.argsort(scores.numpy() * -1, axis=-1)


def main(
    data_split=SplitEnum.dev,
    output_pattern="../predictions/predict.FOLD.baseline-dpr.txt",
):
    data = Data(data_split=data_split)
    data.load()
    data.analyze()
    manager = PredictManager(file_pattern=output_pattern)

    encoder_q = Encoder(name="facebook/dpr-question_encoder-single-nq-base")
    encoder_d = Encoder(name="facebook/dpr-ctx_encoder-single-nq-base")

    r = DprRetriever(encoder_q=encoder_q, encoder_d=encoder_d)
    preds = r.run(data)
    manager.write(preds, data_split)
    if data_split != SplitEnum.test:
        Scorer().run(data.path_gold, manager.make_path(data_split))
        ResultAnalyzer().run(data, preds)


if __name__ == "__main__":
    main()
