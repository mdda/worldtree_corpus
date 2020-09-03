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
nlp = spacy.load("en_core_web_sm", disable=["ner", "deps"])  # "tagger", "parser", 

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
            #doc = spacy.tokens.Doc(nlp.vocab, words=tok_arr, ) # spaces=spaces defaults to all True
            
            # Let's create the raw text we want to process, along with a char-array that
            # tells us which 'cell' (i.e column) we're in
            tok_txt = ' '.join(tok_arr)
            char_idx_arr = [] # identifies the column that each character in raw_txt is in
            for loc, tok in zip(loc_arr, tok_arr):
                # Add a space (one surplus one at the end is not-a-problem)
                #for _ in range(len(tok)+1): 
                #    char_idx_arr.append(loc)
                char_idx_arr.extend( [loc]*(len(tok)+1) )  # Slicker

            tok_txt  = tok_txt.replace("single-celled", "singleXcelled")
            # infrared NOUN
            # two more more materials -> two or more materials

            doc=nlp(tok_txt)
            print(tok_txt)
            #print(char_idx_arr)

            # Plan : 
            #   Get the list of tokens returned by doc, and pop them into respective columns
            #   For each column, read the tokens backwards.
            #      Starting at a Noun, accept NOUN, PROPN, or ADJ until none
            #        Each group is a compound noun : form the actual one with the associated lemmas

            token_arrays_at_columns = [ [] for _ in lex_arr ]
            for i, token in enumerate(doc):
                col_idx = char_idx_arr[ token.idx ]
                print(f"{col_idx:2d} : {token.text.lower():<20s}, {token.lemma_.lower():<20s}, {token.pos_}")
                token_arrays_at_columns[col_idx].append( token )

            # This assumes that tokens is a consecutive list of tokens 
            #   that spacy has processed from a doc
            def get_compound_spans(tokens):
                in_span, current_span, found_spans = False, [], []
                for t in tokens[::-1]:  # Go backwards through the list
                    pos = t.pos_
                    if in_span:
                        if pos in 'NOUN|PROPN|ADJ':
                            current_span.insert(0, t)  # prepend what we found
                        else:
                            found_spans.append(current_span)
                            in_span=False
                    else:
                        if pos in 'NOUN|PROPN|ADJ':  # |ADJ' is experimental
                            in_span=True
                            current_span = [t]
                if in_span:  # We finished without 'closing the current_span'
                    found_spans.append(current_span)
                
                compound_lemmas=[]
                for span in found_spans:
                    compound_lemmas.append( '_'.join(t.lemma_.lower().strip() for t in span) )

                return compound_lemmas

            for col_idx, tokens in enumerate(token_arrays_at_columns):
                # Add the found spans, as lemmas, to the set of lemmas at this location
                for compound_lemma in get_compound_spans(tokens):
                    lex_arr[col_idx].add( compound_lemma )

            # So now go through the 'fields' 
            # - what we want is the 'nodes' associated with every column
            #remove_stop=True
            #remove_punct=True
            #for i, token in enumerate(doc):
            #    col_idx = char_idx_arr[ token.idx ]
            #    print(f"{col_idx:2d} : {token.text.lower():<20s}, {token.lemma_.lower():<20s}, {token.pos_}")
            #    if not token.text.lower() in whitelist_words:  # These get waved through
            #        if token.is_stop and remove_stop:
            #            continue
            #        if token.is_punct and remove_punct:
            #            continue
            #        if len(token.lemma_.strip())==0:
            #            continue # Kill spaces
            #    # Ok, so this is potentially useful as a 'node'
            #    #col_idx = loc_arr[i]
            #    if header_skip[col_idx] or header_fill[col_idx]:
            #        pass
            #    else:
            #        # Add this lemma to the set of lemmas at this location?
            #        lex_arr[col_idx].add( token.lemma_.strip() )

                #_tokens.append(token.text)
                #_lemmas.append(token.lemma_.lower())  
            print("lex_arr", lex_arr)

            #if 'dioxide' in tok_txt: exit(0)

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

    # TODO:
    """
    Noun phrases into compound words...
    Separate commas out in lists in cells
    Think about the word 'and' in lists in cells

    class TxtAndNodes
        txt: str
        nodes: []

    class QuestionAnswer 
        question: TxtAndNodes
        answer  : TxtAndNodes
        wrong   : List[TxtAndNodes]

        explanation_gold: []
        question_statements: [] # Becomes part of initial explanation

    qa_raw = QuestionAnswer()
    qa_enh = qa_raw
    qa_enh = MoveStatmentsFromQuestion(qa_enh)
    qa_enh = MoveAssumptionsFromQuestionToAnswer(qa_enh)
    qa_enh = ResolveExplanationsToSpecifics(qa_enh)

    """
