import os
import warnings

from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

#from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm


def read_explanations(path: str) -> List[Tuple[str, str]]:
    header = []
    uid = None

    df = pd.read_csv(path, sep='\t', dtype=str)

    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid:
                uid = name
        else:
            header.append(name)

    if not uid or len(df) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    return df.apply(lambda r: (r[uid], ' '.join(str(s) for s in list(r[header]) if not pd.isna(s))), 1).tolist()
