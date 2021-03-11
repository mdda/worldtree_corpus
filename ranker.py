import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class CosineRanker:
    def run(self, vecs_q, vecs_e):
        distances = cosine_distances(vecs_q, vecs_e)
        ranking = np.argsort(distances, axis=-1)
        return ranking
