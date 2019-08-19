import os, sys
#import time

#import requests, shutil
import json
import re

from pathlib import Path

import numpy as np
import pandas as pd

"""
path_data = "tg2019task/worldtree_corpus_textgraphs2019sharedtask_withgraphvis"
if not Path(path_data).exists():
    # Download data
    !git clone -q https://github.com/umanlp/tg2019task.git
    !cd tg2019task/ && make dataset
    # Run baseline tfidf (expected MAP: 0.054)
    !cd {path_data} && python ../baseline_tfidf.py annotation/expl-tablestore-export-2017-08-25-230344/tables questions/ARC-Elementary+EXPL-Dev.tsv > predict.txt
    !cd {path_data} && python ../evaluate.py --gold=questions/ARC-Elementary+EXPL-Dev.tsv predict.txt
"""


import nltk

import spacy
nlp = spacy.load("en_core_web_sm")

whitelist_words="""
most least much few
all nothing 
full empty
once used
front back
below above
bottom top
down up
less more
part whole
see move
first one two
show 
something
""".strip().lower().split()
# use make
# 'something' is a bit of a wild-card here...

def convert_texts(texts, remove_stop=True, remove_punct=True):
    def prepreprocess(arr):
        # Sometimes spacy doesn't handle punctuation well eg "work;life"
        # But completely removing all punctuation worsens score
        return [txt.replace(";", "; ") for txt in arr]
    
    tokens, lemmas = [], []
    for doc in nlp.pipe(prepreprocess(texts), disable=["ner", "tagger", "parser"]):
        _tokens, _lemmas = [], []
        for token in doc:
            #print(token.text.lower())
            if not token.text.lower() in whitelist_words:  # These get waved through
                if token.is_stop and remove_stop:
                    continue
                if token.is_punct and remove_punct:
                    continue
                if len(token.lemma_.strip())==0:
                    continue # Kill spaces
            _tokens.append(token.text)
            #_lemmas.append(token.lemma_)  
            _lemmas.append(token.lemma_.strip().lower())  
        tokens.append(_tokens)
        lemmas.append(_lemmas)

    return tokens, lemmas

def get_questions(path_questions, fname):
    df = pd.read_csv(Path(path_questions).joinpath(fname), sep="\t")
    tokens, lemmas = convert_texts(df.Question)
    df["tokens"] = tokens
    df["lemmas"] = lemmas
    print(df.shape)
    return df


def read_explanations_with_permutations(path, uids_existing):  # uids_existing is modified-in-place
    df = pd.read_csv(path, sep='\t')

    header, uid_column, dep_column = [], None, None
    for name in df.columns:
        if name.startswith('[SKIP]'):
            if 'UID' in name and not uid_column:
                uid_column = name # This is the column header
            if 'DEP' in name and not dep_column:
                dep_column = name # This is the column header
        else:
            header.append(name) # These are all those not market '[SKIP]'

    if not uid_column or len(df) == 0:
        print('Possibly misformatted file: ' + path)
        return []

    arr=[]
    for idx in df.index:
        if dep_column is not None:
            dep = df.loc[idx][dep_column]
            if not pd.isna(dep) and len( str(dep).strip() )>0:
                #print(f"Skipping : '{df.loc[idx][dep_column]}' for")
                #print(f"  {df.loc[idx]}")
                if False:  # Actually this hasn't been done properly in the dataset
                    continue
        
        uid_raw = df.at[idx, uid_column]
        if uid_raw in uids_existing:
            print(f"Skipping duplicate uid : '{uid_raw}'")
            continue
        uids_existing.add(uid_raw)
        
        cells, combos, combo_tot = dict(), [], 1
        for h,v in zip(header, df.loc[idx][header]):
            s = '' if pd.isna(v) else v
            options = [ o.strip() for o in str(s).split(';') ]
            options = [ o for o in options if len(o)>0 ]
            if len(options)==0: options=['']
            #print(options)
            cells[h] = options
            combos += [ len(options) ]
            combo_tot *= len(options)  # Count up the number of combos this contributes
            
        for i in range(combo_tot):
            # Go through all the columns, figuring out which combo we're on
            combo, lemmas, residual = [], [], i
            for j,h in enumerate(header):
                # Find the relevant part for this specific combo
                c = cells[h][ residual % combos[j] ] # Works even if only 1 combo
                if len(cells[h])>1: lemmas.append(c) # This is when there are choices
                combo.append( c )
                residual = residual // combos[j]  # TeeHee
            
            # Order : uid, text, musthave, orig
            arr.append( [ 
                f"{uid_raw}_{i}",   # uid_i
                ' '.join( [ c for c in combo if len(c)>0 ] ), # text for this combo
                lemmas,
                ' '.join( [ (f"{{{'; '.join(cells[h])}}}" if len(cells[h])>1 else cells[h][0] ) 
                              for h in header 
                              if len(cells[h][0])>0 ]).strip(), # 'orig' for debug
                os.path.basename(path).replace('.tsv', ''),  # 'table'
                uid_raw, # 'uid_raw'
            ] )
        
    return arr


def get_df_explanations(path_tables):
    explanations, uids_existing = [], set()
    for p in path_tables.iterdir():
        #if 'USEDFOR.tsv' not in str(p): continue
        explanations += read_explanations_with_permutations(str(p), uids_existing) 
        #print(len(uids_existing))  # Check that uids_existing is being modified-in-place
        
    df = pd.DataFrame(explanations, columns=("uid", "text", "musthave", "orig", "table", "uid_raw",))
    
    #print( df[ df.duplicated("uid") ]['uid'] )  # have a look at the problem rows
    #    this problem eliminate in parse above using '[SKIP] DEP' column
    #df = df.drop_duplicates("uid")  # NOO!
    #return df

    tokens, lemmas = convert_texts(df.text)
    #df["tokens"] = tokens
    df["lemmas"] = lemmas
    
    for idx in df.index:
        musthave=df.at[idx, 'musthave']
        for i, musthave_lemma in enumerate(convert_texts(musthave)[1]):
            if len(musthave_lemma)==0:
                print(f"Need to have lemma of '{musthave[i]}' in :\n  '{df.at[idx, 'orig']}'")
    
    print(df.shape)
    return df



def flatten(nested_list):
    return [item for lst in nested_list for item in lst]

def get_flattened_items(dfs, field):
    all_items = []
    for df in dfs:
        all_items.extend(flatten(df[field]))
    return all_items

#all_lemmas = get_flattened_items([df_trn, df_dev, df_test, df_exp], "lemmas")
#80307 original
#80061 all lower-cased, no white-space
# all_tokens = get_flattened_items([df_trn, df_dev, df_test, df_exp], "tokens")
#88152 with 'combos' within exp unravelled
    
#unique_lemmas = sorted(list(set(all_lemmas)))
#node_lemma = unique_lemmas
#n_nodes=len(node_lemma)
#print(n_nodes)
#5262 with upper+lower case
#4754 all lower-cased
#4751 all lower-cased, no white-space
#4758 with whitelist_words
#4766 with whitelist_words including some required to lengthen explanations
#4717 as above with ';'->'; ' fix
#4716 with 'combos' within exp unravelled

#lemma2node = {lemma:idx for idx, lemma in enumerate(node_lemma)}
#def nodes_to_sentence(nodes):
#    return ' '.join([ node_lemma[n] for n in nodes ])
    
#df_lemma = pd.DataFrame({
#    "node": unique_lemmas,
#    "embedding": None,
#})
#df_lemma.sample(5, random_state=42)

def get_node_fns( dfs ):
    all_lemmas = get_flattened_items(dfs, "lemmas")
    unique_lemmas = sorted(list(set(all_lemmas)))
    print(f"Total number of lemmas found : {len(all_lemmas):d}, unique : {len(unique_lemmas)}")

    node_lemma = unique_lemmas
    #n_nodes=len(node_lemma)
    
    lemma2node = {lemma:idx for idx, lemma in enumerate(node_lemma)}
    def nodes_to_sentence(nodes):
        return ' '.join([ node_lemma[n] for n in nodes ])
    
    return node_lemma, lemma2node

def decompose_questions(df, lemma2node):
    def get_nodes(lemmas):
        return [lemma2node[lemma] for lemma in lemmas]

    df['q_lem'], df['a_lem']=None,None
    for prob in df.index:
        multi=re.split(r'\([ABCDEF]\)\s*', df.at[prob,'Question'])
        j_ans = 'ABCDEF'.find(df.at[prob,'AnswerKey'])+1

        q_lem, a_lem=None, []
        _, lemmas = convert_texts(multi)
        for j,ls in enumerate(lemmas):
            #print(ls)
            ids = get_nodes(ls)
            #print(ls, ids, j_ans)
            if 0==len(ids):
                # Just show that there's a problem (fortunately, no correct answer has 0 lemma terms)
                print(prob, multi, lemmas, j, j_ans, ids) 
            if 0==j:
                q_lem=ids # This is the question
            else:
                if j==j_ans:  # This is the correct answer (reorder to first in list)
                    a_lem.insert(0, ids)
                else:           # Wrong answers come after correct one
                    a_lem.append(ids)

        if False:
            print(q_lem)
            for a in a_lem:
                print("   ", a)
                
        df.at[prob,'q_lem']=q_lem
        df.at[prob,'a_lem']=a_lem
        

def add_tfidf_questions(df, node_lemma):
    for prob in df.index:
        question_with_ans = df.at[prob,'q_lem'] + df.at[prob,'a_lem'][0]
        df.at[prob,'q_tfidf']=' '.join(
            [ node_lemma[n] for n in question_with_ans ]
        )
  
def add_tfidf_explanation(dfe, node_lemma):
    for e in dfe.index:
        ex_nodes = dfe.at[e,'nodes']
        dfe.at[e,'e_tfidf']=' '.join([ node_lemma[n] for n in ex_nodes ])


def add_gold_explanation_idx(df, df_exp):
    df['ex_gold']=None
    df['ex_reason']=None
    for prob in df.index:
        exs_str = df.at[prob, 'explanation']
        ex_arr, reason_arr = [],[]
        if type(exs_str) is str: 
            for ex_str in exs_str.split(' '):
                uid, reason = ex_str.split('|')
                ex_uid_rows = df_exp[ df_exp['uid_raw'] == uid ]
                exs_arr=[]
                for i in range(len(ex_uid_rows)):
                    exs_arr.append( df_exp.index.get_loc(ex_uid_rows.iloc[i].name) ) 
                if 0==len(exs_arr):
                    print("Missing ID '%s' in '%s'" % (uid, exs_str,))
                    continue # missing uid, somehow
                #print(exs_arr)
                ex_arr.append(exs_arr)
                reason_arr.append(reason)
        df.at[prob, 'ex_gold']=ex_arr # Each explanation is an array of uid_raw aliases
        df.at[prob, 'ex_reason']=reason_arr  # Only for those found
        
