import os
import warnings

from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple

import csv
import numpy as np
#import pandas as pd

#from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "deps"])  # "tagger", 


# Type-definitions
UID=str

class Statement(BaseModel):
    uid_base:UID
    uid:     UID
    table:   str
    raw_txt: str


# Hard-coded whitelist (!)
whitelist_words="""
most least much few
all nothing 
full empty
once used
front back
see move
""".strip().lower().split()


def read_explanation_file(path: str, table_name: str) -> List[Tuple[UID, Statement]]:
    header, fields, rows, uid_col = [], dict(), [], None

    with open(path, 'rt') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            if i==0: 
                header=row
                header_skip=[False]*len(header)
                header_fill=[True] *len(header)
            else:    
                rows.append(row)

    for i, col_name in enumerate(header):
        if col_name.startswith('[SKIP]'):
            header_skip[i]=True
            # Find first column name that contains '[SKIP] ... UID ...'
            if 'UID' in col_name and uid_col is None:
                uid_col = i
        else:
            if not col_name.startswith('[FILL]'):
                header_fill[i]=False
                fields[col_name]=i  # Only 'proper' ones
    #print(fields)

    if uid_col is None or len(rows) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []

    statements=[]
    for i, row in enumerate(rows):
        #print(header_skip)
        #print(header_fill)
        # Convert this single row into the combo rows
        combo_row = [ ([''] if header_skip[i] else cell.split(';')) for i, cell in enumerate(row) ]
        #print( combo_row )
        n_combos = np.prod([len(combo) for combo in combo_row])

        #combos = [ row ]  # The original version
        combos = [ [cell.replace(';', ' ;') for cell in row] ]  # The ~original version
        if n_combos==1:
            # If there are no choices, then n_combos==1
            #   Don't add anything extra - already have the original
            pass
        else:
            # Need to create all combos via products, etc
            #   but if there are choices, also keep the combo one (with ';' in it)
            for j in range(n_combos): # generating the j-th combo
                combo, j_iter=[], j
                for k, cell in enumerate(combo_row):
                    len_cell=len(cell)
                    # This modulus/dividing thing is pure CS
                    combo.append( cell[j_iter % len_cell].strip() )
                    j_iter //= len_cell
                combos.append( combo )
                #print("   ", combo)

        # Ok, so now have the expanded list of combos for this line
        for combo_idx, combo_row in enumerate(combos):
            txt_arr, lex_arr = [], []  # Store the actual text in each relevant column
            loc_arr, tok_arr = [], []
            for i,s in enumerate(combo_row):
                txt  = s.strip()
                if header_skip[i]: txt=''
                txt_arr.append(txt)
                lex_arr.append( set() )

                if len(txt)>0:
                    toks = txt.split(' ')
                    locs = [i]*len(toks)

                    tok_arr.extend(toks)
                    loc_arr.extend(locs)

            raw_txt = ' '.join( txt_arr )
            if True:
                print()
                print("txt_arr : "+(' | '.join( txt_arr )))
                #raw_tok = ' | '.join( tok_arr )
                #print("raw_tok : "+raw_tok)  # looks fine
                #raw_loc = ' | '.join( str(i) for i in loc_arr )
                #print("raw_loc : "+raw_loc)  # looks fine

            # https://spacy.io/api/doc : Construct a doc word-by-word (preserves positions)
            # Potentially better : https://explosion.ai/blog/spacy-v2-pipelines-extensions
            doc = spacy.tokens.Doc(nlp.vocab, words=tok_arr, ) # spaces=spaces defaults to all True
            remove_stop=True
            remove_punct=True

            # So now go through the 'fields' 
            # - what we want is the 'nodes' associated with every column
            for i, token in enumerate(doc):
                #print(token.text.lower(), token.lemma_.lower())
                if not token.text.lower() in whitelist_words:  # These get waved through
                    if token.is_stop and remove_stop:
                        continue
                    if token.is_punct and remove_punct:
                        continue
                    if len(token.lemma_.strip())==0:
                        continue # Kill spaces
                # Ok, so this is potentially useful as a 'node'
                col_idx = loc_arr[i]
                if header_skip[col_idx] or header_fill[col_idx]:
                    pass
                else:
                    # Add this lemma to the set of lemmas at this location?
                    lex_arr[col_idx].add( token.lemma_.strip() )

                #_tokens.append(token.text)
                #_lemmas.append(token.lemma_.lower())  
            print("lex_arr", lex_arr)

            uid = row[uid_col]
            s = Statement(
                uid_base= uid,
                uid     = uid if n_combos==1 else f"{uid}_{combo_idx}",
                table   = table_name, 
                raw_txt = raw_txt,
            )
            statements.append(s)
            #print()
            

        if i>50: break
    return statements

if '__main__' == __name__:
    #os.path.join(path, file)
    table='MADEOF'
    ex_file=os.path.join('../tg2020task/tables', f'{table}.tsv')
    explanations = read_explanation_file(ex_file, table)
    #print(explanations)
