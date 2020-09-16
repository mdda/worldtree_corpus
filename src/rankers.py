from typing import List, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances
from torch import Tensor
from tqdm import tqdm


def deduplicate(items: list) -> list:
    seen = set()
    output = []
    for x in items:
        if x not in seen:
            seen.add(x)
            output.append(x)
    return output


class Ranker(BaseModel):
    def run(self, vecs_q: csr_matrix, vecs_s: csr_matrix) -> np.ndarray:
        distances: np.ndarray = cosine_distances(vecs_q, vecs_s)
        ranking: np.ndarray = np.argsort(distances, axis=-1)
        return ranking


class StageRanker(Ranker):
    # Dev MAP: 0.3816
    num_per_stage: List[int] = [25, 100]
    scale: float = 1.0

    def recurse(
        self,
        vec_q: csr_matrix,
        vecs_s: csr_matrix,
        indices_s: np.ndarray,
        num_per_stage: List[int],
        num_accum: int = 0,
    ) -> List[int]:
        num_s = vecs_s.shape[0]
        assert num_s == len(indices_s)
        if num_s == 0:
            return []

        num_keep = num_s
        if num_per_stage:
            num_keep = num_per_stage.pop(0)
        num_next = max(num_s - num_keep, 0)

        distances: np.ndarray = cosine_distances(vec_q, vecs_s)[0]
        assert distances.shape == (num_s,)
        rank = np.argsort(distances)

        vecs_keep = vecs_s[rank][:num_keep]
        indices_keep = indices_s[rank][:num_keep]
        vec_new = np.max(vecs_keep, axis=0)
        vec_new = vec_new / (self.scale ** num_accum)
        if isinstance(vec_q, csr_matrix):
            vec_q = vec_q.maximum(vec_new)
        else:
            vec_q = np.maximum(vec_q, vec_new)
        num_accum += num_keep

        if num_next == 0:
            vecs_s = np.array([])
            indices_s = np.array([])
        else:
            vecs_s = vecs_s[rank][-num_next:]
            indices_s = indices_s[rank][-num_next:]

        return list(indices_keep) + self.recurse(
            vec_q, vecs_s, indices_s, num_per_stage, num_accum,
        )

    def run(self, vecs_q: csr_matrix, vecs_s: csr_matrix) -> np.ndarray:
        num_q = vecs_q.shape[0]
        num_s = vecs_s.shape[0]
        ranking = np.zeros(shape=(num_q, num_s), dtype=np.int)
        for i in tqdm(range(num_q)):
            ranking[i] = self.recurse(
                vecs_q[[i]], vecs_s, np.arange(num_s), list(self.num_per_stage)
            )
        return ranking


class IterativeRanker(Ranker):
    max_seq_len: int = 128
    top_n: int = 1
    scale: float = 1.25

    def recurse(
        self, vec_q: csr_matrix, vecs_s: csr_matrix, indices: List[int],
    ):
        if len(indices) >= self.max_seq_len:
            return

        distances: np.ndarray = cosine_distances(vec_q, vecs_s)[0]
        assert distances.shape == (vecs_s.shape[0],)
        rank = np.argsort(distances)

        seen = set(indices)
        count = 0
        for i in rank:
            if count == self.top_n:
                break

            if i not in seen:
                vec_new = vecs_s[i] / (self.scale ** len(indices))
                if isinstance(vec_q, csr_matrix):
                    vec_q = vec_q.maximum(vec_new)
                else:
                    vec_q = np.maximum(vec_q, vec_new)
                indices.append(i)
                count += 1
                self.recurse(vec_q, vecs_s, indices)

    def run(self, vecs_q: csr_matrix, vecs_s: csr_matrix) -> np.ndarray:
        ranking = super().run(vecs_q, vecs_s)
        rank_old: np.ndarray
        for i, rank_old in tqdm(enumerate(ranking), total=len(ranking)):
            rank_new = []
            self.recurse(vecs_q[[i]], vecs_s, rank_new)
            assert rank_new
            rank_new = rank_new + list(rank_old)
            rank_new = deduplicate(rank_new)
            ranking[i] = rank_new
        return ranking


class WordEmbedRanker(Ranker):
    device: str = "cuda"

    def pad_embeds(self, embeds: List[List[Tensor]]) -> Tuple[Tensor, Tensor]:
        lengths = [len(_) for _ in embeds]
        dim = len(embeds[0][0])
        print("WordEmbedRanker: Embeds dim:", dim)
        x = torch.zeros(len(embeds), max(lengths), dim, device=self.device)
        mask = torch.zeros(len(embeds), max(lengths), device=self.device)

        for i, lst in enumerate(embeds):
            mask[i, : lengths[i]] = 1.0
            if lst:
                x[i, : lengths[i], :] = torch.stack(lst)

        return x, mask

    @staticmethod
    def run_max_sim(q: Tensor, x: Tensor, mask: Tensor) -> Tensor:
        len_q, dim_q = q.shape
        num_x, len_x, dim_x = x.shape
        assert dim_q == dim_x
        q = torch.unsqueeze(q, dim=0)
        q = q.repeat(num_x, 1, 1)
        assert tuple(q.shape) == (num_x, len_q, dim_q)

        lengths = torch.sum(mask, dim=1)
        mask = torch.unsqueeze(mask, dim=1)  # num_x, 1, len_x

        x = torch.bmm(q, x.permute(0, 2, 1))  # num_x, len_q, len_x
        x = torch.mul(x, mask)
        x = torch.max(x, dim=2).values
        x = torch.sum(x, dim=1)
        assert x.shape == lengths.shape
        x = torch.div(x, lengths)
        return x

    def run(
        self, embeds_q: List[List[Tensor]], embeds_s: List[List[Tensor]]
    ) -> np.ndarray:
        x, mask = self.pad_embeds(embeds_s)
        scores = torch.zeros(len(embeds_q), len(embeds_s), device=self.device)

        for i, lst in tqdm(enumerate(embeds_q), total=len(embeds_q)):
            q = torch.stack(lst).to(self.device)
            scores[i] = self.run_max_sim(q, x, mask)

        scores_numpy = scores.cpu().numpy()
        ranking: np.ndarray = np.argsort(scores_numpy * -1, axis=-1)
        return ranking
