import os
import warnings

from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple

import csv
import numpy as np
#import pandas as pd

#from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

UID=str

class Statement(BaseModel):
    uid_base:UID
    uid:     UID
    table:   str
    raw_txt: str


#def read_explanations(path: str) -> List[Tuple[str, str]]:
def read_explanations(path: str) -> List[Tuple[UID, Statement]]:
    header, fields, rows = [], dict(), []
    uid_col = None

    #df = pd.read_csv(path, sep='\t', dtype=str)
    with open(path, 'rt') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            if i==0: header=row
            else:    rows.append(row)

    for i, col_name in enumerate(header):
        if col_name.startswith('[SKIP]'):
            # Find first column name that contains '[SKIP] ... UID ...'
            if 'UID' in col_name and not uid_col:
                uid_col = col_name
        else:
            if not col_name.startswith('[FILL]'):
                fields[col_name]=i  # Only 'proper' ones

    if not uid_col or len(rows) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    #def raw_txt(r):
        #return ' '.join( str(s) for s in list(r[header]) if not pd.isna(s) )
    #    return ' '.join( str(s) for s in list(r[header]) if not pd.isna(s) )
    #df.apply(lambda r: (r[uid], raw_txt(r)), 1).tolist()

    #print(fields)
    for i, row in enumerate(rows):
        raw_txt = ' '.join( str(s) for i,s in enumerate(row) 
          if len(s)>0 and header[i]!=uid_col and not s.startswith('#') )

        # Convert this single row into the combo rows
        combo_row = [ c.split(';') for c in row ]
        n_combos = np.prod([len(combo) for combo in combo_row])
        #if n_combos==1:
        #    combos = [ row ]  # Nothing changed - only 1 combo element in each position
        #else:
        if True:
            print( combo_row )
            # Need to create a combo via products, etc
            combos = []
            for j in range(n_combos): # generating the j-th combo
                combo, j_iter=[], j
                for k, c in enumerate(combo_row):
                    len_c=len(c)
                    # This modulus/dividing thing is pure CS
                    combo.append( c[j_iter % len_c].strip() )
                    j_iter //= len_c
                combos.append( combo )
                print("   ", combo)

        if i>50: break

# nlp = spacy.load("en_core_web_sm")

if '__main__' == __name__:
    #os.path.join(path, file)
    ex_file=os.path.join('../tg2020task/tables', 'MADEOF.tsv')
    explanations = read_explanations(ex_file)
    #print(explanations)
