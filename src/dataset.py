import os
import warnings

from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple, Set

import re, csv, json
import numpy as np
#import pandas as pd

#from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "deps"])  # "tagger", "parser", 

TASK_BASE = '../tg2020task'
RDAI_BASE = '../data/'


# Type-definitions
UID=str
Keywords=Set[str]

class Statement(BaseModel):
    uid_base:UID
    uid:     UID
    table:   str

    hdr_arr: List[str]
    txt_arr: List[str]
    #tok_arr: List[str]
    keyword_arr: List[Keywords]

    raw_txt: str
    keywords: Keywords

class TxtAndKeywords(BaseModel):
    txt: str
    keywords: List[Keywords]

class QuestionAnswer(BaseModel):
    question: TxtAndKeywords
    answer  : TxtAndKeywords
    wrong   : List[TxtAndKeywords]

    explanation_gold: List[Statement] = []
    # Becomes part of initial explanation :
    question_statements: List[Statement] = [] 



# Hard-coded whitelist (!)
UNUSED_whitelist_words="""
most least much few
all nothing 
full empty
once used
front back
see move
""".strip().lower().split()

hyphenations=None
def fix_hyphenations(txt):
    global hyphenations

    hyphenated = re.findall(r'\w+\-[\w\-]+', txt)
    if len(hyphenated)==0:
        return txt # Leave quickly

    # Read and cache the hyphenations.csv file
    if hyphenations is None:
        hyphenations=dict()
        with open(os.path.join(RDAI_BASE, 'hyphenations.csv')) as f:
            for l in f.readlines():
                l=l.strip()
                if len(l)==0 or l.startswith('#'):
                    continue
                hyphenations[l] = l.replace('-', '+')

    for h in hyphenated:
        print(f"DEBUG:hyphenated:{h.strip()}:{txt.strip()}")
        # Mutating value of txt : Oh No!
        txt = txt.replace(h, hyphenations.get(h, h))
    return txt    

keyword_relabel=None  # Will be a dictionary map to sets of replacement kewords
def fix_keyword_set(kws):
    global keyword_relabel

    # Read and cache the keyword_relabel.txt file
    if keyword_relabel is None:
        keyword_relabel=dict()
        with open(os.path.join(RDAI_BASE, 'keyword_relabel.txt')) as f:
            for l in f.readlines():
                l=l.strip()
                if len(l)==0 or l.startswith('#'):
                    continue
                kw_from, kw_to = l.split(':')
                for kw_old in kw_from.split(','):
                    for kw_new in kw_to.split(','):
                        if not kw_old in keyword_relabel:
                            keyword_relabel[kw_old]=set()
                        keyword_relabel[kw_old].add(kw_new)

    # jq -c .raw_txt,.keywords data/statements.jsonl | grep -A0 -B1 'chemical_bond_energy'
    kw_replaced=set()
    for kw in kws:
        if kw in keyword_relabel:
            print(f"DEBUG:keyword_replace:{kw}->{keyword_relabel[kw]}")
            kw_replaced.update(keyword_relabel[kw]) # This is a set of replacements
        else:
            # This just copies the old value across
            kw_replaced.add(kw)
    return kw_replaced

# This assumes that tokens is a consecutive list of tokens 
#   that spacy has processed from a doc
def extract_keywords(spacy_tokens, require_keywords=True) -> Keywords:
    in_span, current_span, found_spans = False, [], []
    for t in spacy_tokens[::-1]:  # Go backwards through the list
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
    
    if len(found_spans)==0 and require_keywords:
        # Be a bit looser... Take the VERB if any
        priority=['VERB', 'ADV']
        for p in priority:
            for t in spacy_tokens[::-1]:  # Go backwards through the list
                if t.pos_ == p:
                    # This isn't really a span.. == LAZY for now
                    found_spans.append([t])
            if len(found_spans)>0: 
                break # Stop going through list once something found
            
    #keywords=[]
    #for span in found_spans:
    #    keywords.append( '_'.join(t.lemma_.lower().strip() for t in span) )
    keywords=set()
    for span in found_spans:
        keyword = '_'.join(t.lemma_.lower().strip() for t in span)
        keywords.add( keyword )
    return fix_keyword_set(keywords)


def read_explanation_file(path: str, table_name: str) -> List[Statement]:
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
    for row_i, row in enumerate(rows):
        #print(header_skip)
        #print(header_fill)
        # Convert this single row into the combo rows
        combo_row = [ ([''] if header_skip[col_i] else cell.split(';')) for col_i, cell in enumerate(row) ]
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
            txt_arr, keyword_arr = [], []  # Store the actual text in each relevant column
            hdr_arr, loc_arr, tok_arr = [], [], []
            for col_i,s in enumerate(combo_row):
                txt  = s.strip()
                if header_skip[col_i]: txt=''
                txt_arr.append(txt)
                keyword_arr.append( set() )

                hdr_arr.append('' if header_fill[col_i] else header[col_i])
                if len(txt)>0:
                    toks = txt.split(' ')
                    locs = [col_i]*len(toks)

                    tok_arr.extend(toks)
                    loc_arr.extend(locs)

            #raw_txt = ' '.join( txt_arr )
            raw_txt = ' '.join( t for t in txt_arr if len(t)>0 )

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

            tok_txt = fix_hyphenations(tok_txt)

            # infrared NOUN

            # two more more materials -> two or more materials
            # distructive potential of hurricanes -> destructive potential of hurricanes
            # cooler than yellow-dwaf stars -> cooler than yellow-dwarf stars

            doc=nlp(tok_txt)

            if True:
                print()
                print("txt_arr : "+(' | '.join( txt_arr )))
                #raw_tok = ' | '.join( tok_arr )
                #print("raw_tok : "+raw_tok)  # looks fine
                #raw_loc = ' | '.join( str(i) for i in loc_arr )
                #print("raw_loc : "+raw_loc)  # looks fine
                #print(char_idx_arr)
                print(tok_txt)

            # Plan : 
            #   Get the list of tokens returned by doc, and pop them into respective columns
            #   For each column, read the tokens backwards.
            #      Starting at a Noun, accept NOUN, PROPN, or ADJ until none
            #        Each group is a compound noun (==Keyword) 
            #        : form the actual one with the associated lemmas

            token_arrays_at_columns = [ [] for _ in keyword_arr ]
            for tok_i, token in enumerate(doc):
                col_idx = char_idx_arr[ token.idx ]
                print(f"{col_idx:2d} : {token.text.lower():<20s}, {token.lemma_.lower():<20s}, {token.pos_}")
                token_arrays_at_columns[col_idx].append( token )

            for col_idx, tokens in enumerate(token_arrays_at_columns):
                if header_fill[col_idx]: continue  # no keywords in [FILL] columns
                # Add the found spans, converted to keywords, 
                #   to the set of keywords at this location
                keywords = extract_keywords(tokens)
                keyword_arr[col_idx].update( keywords )
            print("keyword_arr", keyword_arr)

            #for s in keyword_arr:
            #    print(list(s))
            #print(set.union(*keyword_arr))

            uid = row[uid_col]
            s = Statement(
                uid_base= uid,
                uid     = f"{uid}_{combo_idx}" if n_combos>1 else uid,
                table   = table_name, 

                hdr_arr = hdr_arr,
                txt_arr = txt_arr,
                #tok_arr = tok_arr,
                keyword_arr = keyword_arr,

                raw_txt = raw_txt,
                keywords = set.union(*keyword_arr),
            )
            statements.append(s)
            #print()
            #if 'singleXcelled' in raw_txt: exit(0)
            #if 'single+celled' in raw_txt: exit(0)
            #if 'single' in raw_txt: exit(0)

        #if row_i>5: break
    return statements

if '__main__' == __name__:
    statements_file = os.path.join(RDAI_BASE, 'statements.jsonl')

    if not os.path.isfile(statements_file):
        with open(os.path.join(TASK_BASE, 'tableindex.txt'), 'rt') as index:
            tables = [l.replace('.tsv', '').strip() for l in index.readlines()]

        statements_all=[]
        for table in tables:
            #table='MADEOF'
            ex_file=os.path.join(TASK_BASE, 'tables', f'{table}.tsv')
            statements = read_explanation_file(ex_file, table)
            statements_all.extend( statements )
        
        #print(explanations)
        #print(statements[12].json())

        # Generates an 8Mb file 
        with open(statements_file, 'wt') as f:
            for s in statements_all:
                f.write( s.json() )
                f.write('\n')

    # Now load in the statements file
    statements = []
    with open(statements_file, 'rt') as f:
        for l in f.readlines():
            #print(json.loads(l))
            s = Statement.parse_raw( l )
            statements.append( s )

    #print(len(statements))  # 13K in total (includes COMBOs)
    #print(statements[123])

    # Ok, so let's look at the unique Keywords
    keyword_counts = dict()
    for s in statements:
        for k in s.keywords:
            if not k in keyword_counts:
                keyword_counts[k]=0
            keyword_counts[k]+=1
    for k,v in sorted(keyword_counts.items()):
        print(f"{v:4d} : {k}")
    # jq -c . data/statements.jsonl | grep termite

    """
    # TODO:
    DONE : Noun phrases into compound words =Keywords (~works)
    DONE : Separate commas out in lists in cells (works)
    DONE : Think about the word 'and' in lists in cells (works)
    DONE : Force extraction of at least 1 keyword from a token span (input flag)
    DONE : Look for badly hyphenated word '-' and fix  (exceptions file into repo)
    DONE : Look for badly transferred keywords ({'red_light'} should be {'red', 'light'}) and fix (exceptions file into repo)
    More fix-ups for keyword relabelling (ongoing)

    Do keywords and other basic preproc on Q&A datasets
    Do some of the fancier preproc ideas on Q&A datasets

    qa_raw = QuestionAnswer()
    qa_enh = qa_raw
    qa_enh = MoveStatmentsFromQuestion(qa_enh)
    qa_enh = MoveAssumptionsFromQuestionToAnswer(qa_enh)
    qa_enh = ResolveExplanationsToSpecifics(qa_enh)

    Make basic graph from Keyword interconnects of Statements
    Find shortest path from Q->A via https://en.wikipedia.org/wiki/Dijkstra's_algorithm


    """
