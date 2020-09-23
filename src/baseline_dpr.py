from typing import Optional, List

import numpy as np
import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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

from baseline_retrieval import (
    Retriever,
    Data,
    PredictManager,
    Scorer,
    ResultAnalyzer,
    SpacyProcessor,
    TextProcessor,
)
from dataset import TxtAndKeywords
from extra_data import SplitEnum
from vectorizers import TruncatedSVDVectorizer, BM25Vectorizer


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

    def fit(self, texts: List[str]):
        pass

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


class SentenceTransformerEncoder(Encoder):
    name: str = "distilbert-base-nli-stsb-mean-tokens"

    def run(self, texts: List[str]) -> Tensor:
        model = SentenceTransformer(self.name)
        x = model.encode(
            texts,
            batch_size=self.bs,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=True,
        )
        return x


class TruncatedSVDEncoder(Encoder):
    name: str = ""
    vectorizer: TfidfVectorizer = TruncatedSVDVectorizer(
        BM25Vectorizer(), n_components=768
    )
    preproc: TextProcessor = SpacyProcessor()

    def process_texts(self, texts: List[str]) -> List[str]:
        return [self.preproc.run(TxtAndKeywords(raw_txt=t)) for t in texts]

    def fit(self, texts: List[str]):
        texts = self.process_texts(texts)
        self.vectorizer.fit(texts)

    def run(self, texts: List[str]) -> Tensor:
        texts = self.process_texts(texts)
        x = self.vectorizer.transform(texts)
        return torch.from_numpy(x)


class DprRetriever(Retriever):
    encoder_q: Encoder
    encoder_d: Optional[Encoder]
    do_normalize: bool = True

    @staticmethod
    def normalize(x: Tensor) -> Tensor:
        assert x.ndim == 2
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = torch.div(x, norm)
        return x

    def rank(self, queries: List[str], statements: List[str]) -> np.ndarray:
        if self.encoder_d is None:
            self.encoder_d = self.encoder_q

        self.encoder_q.fit(queries + statements)
        self.encoder_d.fit(queries + statements)
        vecs_q = self.encoder_q.run(queries)
        vecs_d = self.encoder_d.run(statements)
        if self.do_normalize:
            vecs_q, vecs_d = self.normalize(vecs_q), self.normalize(vecs_d)

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

    retrievers = [
        DprRetriever(
            encoder_q=Encoder(name="facebook/dpr-question_encoder-single-nq-base"),
            encoder_d=Encoder(name="facebook/dpr-ctx_encoder-single-nq-base"),
            do_normalize=False,
        ),  # Dev MAP=0.3127
        DprRetriever(encoder_q=SentenceTransformerEncoder()),  # Dev MAP=0.3243
        DprRetriever(
            encoder_q=SentenceTransformerEncoder(
                name="roberta-large-nli-stsb-mean-tokens"
            )
        ),  # Dev MAP=0.3150
        DprRetriever(encoder_q=TruncatedSVDEncoder()),  # Dev MAP=0.3847
    ]

    r = retrievers[-1]
    preds = r.run(data)
    manager.write(preds, data_split)
    if data_split != SplitEnum.test:
        Scorer().run(data.path_gold, manager.make_path(data_split))
        ResultAnalyzer().run(data, preds)


if __name__ == "__main__":
    main()
