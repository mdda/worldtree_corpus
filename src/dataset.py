import os, sys
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
    raw_txt: str
    keywords: Keywords=set()

class ExplanationUsed(BaseModel):
    uid:   UID
    reason: str

class QuestionAnswer(BaseModel):
    question_id: UID

    question: TxtAndKeywords
    # Correct answer is at answers[0]
    answers  : List[TxtAndKeywords]

    explanation_gold: List[ExplanationUsed]
    # Becomes part of initial explanation :
    question_statements: List[Statement] = [] 

    flags:str="success"

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
    found_spans = []
    # X|INTJ are here because they pick up useful fragments (not exactly correctly)
    # 'something'=PRON ... to be considered
    pos_set = set('NOUN|PROPN|ADJ|VERB|ADV|NUM|X|INTJ'.split('|'))
    for t in spacy_tokens:
        pos = t.pos_
        #if pos=='X':
        #    print(f"DEBUG:X:{t.text}", spacy_tokens)
        if pos in pos_set: # or t.lemma_=='something':
            found_spans.append([t])
            
    if len(spacy_tokens)>0 and len(found_spans)==0 and require_keywords:
        if False:
            s_arr=[]
            for t in spacy_tokens:
                s_arr.append(f"{t.pos_}:{t.text}={t.lemma_}")
            print(f"DEBUG:no-keywords:{','.join(s_arr)}")
        if len(spacy_tokens)==1:
            # There's only one word in the span : Use it!
            found_spans.append([spacy_tokens[0]])

    keywords=set()
    for span in found_spans:
        keywords.add( span[0].lemma_.lower().strip() )
    return fix_keyword_set(keywords)

# This assumes that tokens is a consecutive list of tokens 
#   that spacy has processed from a doc
def extract_keywords_complex(spacy_tokens, require_keywords=True) -> Keywords:
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

    if len(found_spans)==0 and require_keywords:
        if len(spacy_tokens)==1:
            # There's only one word in the span : Use it!
            found_spans.append([spacy_tokens[0]])
            
    #keywords=[]
    #for span in found_spans:
    #    keywords.append( '_'.join(t.lemma_.lower().strip() for t in span) )
    keywords=set()
    for span in found_spans:
        keyword = '_'.join(t.lemma_.lower().strip() for t in span)
        keywords.add( keyword )
        if True:  # Add the last word in the span as a keyword too
            keyword = span[-1].lemma_.lower().strip()
            keywords.add( keyword )
    return fix_keyword_set(keywords)


def read_explanation_file(path: str, table_name: str) -> List[Statement]:
    header, fields, rows, uid_col, dep_col = [], dict(), [], None, None

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
            if 'DEP' in col_name and dep_col is None:
                dep_col = i
        else:
            if not col_name.startswith('[FILL]'):
                header_fill[i]=False
                fields[col_name]=i  # Only 'proper' ones
    #print(fields)

    if uid_col is None or len(rows) == 0:
        warnings.warn('Possibly misformatted file: ' + path)
        return []
    if dep_col is None:
        warnings.warn('NO DEP COLUMN : ' + path)

    statements=[]
    for row_i, row in enumerate(rows):
        #print(header_skip)
        #print(header_fill)
        if dep_col is not None:
            if len(row[dep_col].strip())>0:
                pass      # Not ignoring lines : 
                # {'statements': 13043, 'questions': 496}
                # baseline_retreival MAP:  0.437252217579726
                print(f"Skipping row '{row[dep_col]}':\n", row)
                continue  # With ignoring these lines : 
                # {'statements': 12045, 'questions': 496}
                # baseline_retreival MAP:  0.4586891839624746

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
            #? predicting weather requires studying weater -> predicting weather requires studying weather

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
                uid     = uid if combo_idx==0 else f"{uid}_{combo_idx}",
                uid_base= uid,
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
            #if uid=='1a4b-82df-0613-5268': exit(0)

        #if row_i>5: break
    return statements

def load_statements(regenerate=False):
    statements_file = os.path.join(RDAI_BASE, 'statements.jsonl')

    if regenerate or not os.path.isfile(statements_file):
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

    # Load in the preprocessed statements file
    statements = []
    with open(statements_file, 'rt') as f:
        for l in f.readlines():
            #print(json.loads(l))
            s = Statement.parse_raw( l )
            statements.append( s )

    return statements

def print_keyword_counts(keyword_counts:Dict[str,int])->None:
    for k,v in sorted(keyword_counts.items()):
        print(f"{v:4d} : {k}")
    # jq -c . data/statements.jsonl | grep termite

def get_keyword_counts_from_statements(statements:List[Statement])->Dict[str,int]:
    keyword_counts = dict()
    for s in statements:
        for k in s.keywords:
            if not k in keyword_counts:
                keyword_counts[k]=0
            keyword_counts[k]+=1
    return keyword_counts


def read_qanda_file(version:str) -> List[QuestionAnswer]:
    qanda_file=os.path.join(TASK_BASE, f'questions.{version}.tsv')

    header, rows = [], []
    with open(qanda_file, 'rt') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            if i==0: 
                header=row
            else:    
                rows.append(row)
    print(f"Read {len(rows)} questions from {version}")

    cols=[ header.index(col) for col in "QuestionID,question,AnswerKey,explanation,flags".split(',') ]

    qa_arr=[]
    for row in rows:
        q_id, q_txt, a_label, ex_gold, flags = [row[col] for col in cols]

        multi=re.split(r'\s*(\([A-F1-6]\))\s*', q_txt)
        #print(multi) # question itself is in [0]

        pos_label = multi.index(f'({a_label})')
        if pos_label<=0:
            print(f"BAD LABEL '{a_label}' in question {row[col_id]}")
            exit(0)

        answer, wrong = None, []
        for i in range(1,len(multi), 2):
            txt = multi[i+1]
            if i==pos_label:
                answer = txt
            else:
                wrong.append(txt)

        # Reorder, so that correct answer is always '0' in the list
        answers=[ TxtAndKeywords(raw_txt=answer) ]      
        answers.extend(
            [ TxtAndKeywords(raw_txt=txt) for txt in wrong ] 
        )

        explanations=[]
        for ex_reason in ex_gold.split(' '):
            if len(ex_reason)==0: 
                #print(version, q_id, ex_reason)
                break  # Mercury_7221305 (and whole of test set)
            statement_uid, reason = ex_reason.split('|')
            explanations.append(
                ExplanationUsed(uid=statement_uid, reason=reason)
            )

        qa = QuestionAnswer(
                question_id=q_id,
                question= TxtAndKeywords(raw_txt=multi[0]),
                answers = answers,
                explanation_gold = explanations,
                flags = flags.lower(),
            )
        qa_arr.append( qa )

    #print(len(qa_arr))
    return qa_arr

def add_keywords_direct(tk:TxtAndKeywords)->None:
    tok_txt = fix_hyphenations(tk.raw_txt)
    doc=nlp(tok_txt)
    tk.keywords = extract_keywords([ t for t in doc]) 
    #return tk

def load_qanda(version:str, regenerate=False)-> List[QuestionAnswer]:
    qanda_cache_file = os.path.join(RDAI_BASE, f'questions.{version}.jsonl')
    if regenerate or not os.path.isfile(qanda_cache_file):
        qanda = read_qanda_file(version)

        for qa in qanda:
            # Changes underlying TxtAndKeywords internally
            add_keywords_direct(qa.question)
            for a in qa.answers:
                add_keywords_direct(a)

        qanda_all = qanda
        # Save in qanda_cache_file
        with open(qanda_cache_file, 'wt') as f:
            for qa in qanda_all:
                f.write( qa.json() )
                f.write('\n')

    # Load in the preprocessed qanda file
    qa_arr = []
    with open(qanda_cache_file, 'rt') as f:
        for l in f.readlines():
            #print(json.loads(l))
            qa = QuestionAnswer.parse_raw( l )
            qa_arr.append( qa )
    return qa_arr

def qa_display(qa:QuestionAnswer,qa_ex_statements:List[Statement])->None:
    print(qa.question.raw_txt)
    for i,a in enumerate(qa.answers):
        print(f"   {'*' if i==0 else '-'}) : {a.raw_txt}")
    for i,s in enumerate(qa.question_statements):
        print(f"b{i:1x} : [question txt]: {s.raw_txt}")
    for i,s in enumerate(qa_ex_statements):
        print(f"e{i:1x} : {'['+qa.explanation_gold[i].reason+']':<14s}: {s.raw_txt}")

def qa_display_keywords(qa:QuestionAnswer, statements:List[Statement])->None:
    # Get GUID -> statement dict
    guid_to_statement={s.uid:s for s in statements}
    # Look up the statements for the qa's gold explanation
    qa_ex_statements=[ 
        guid_to_statement[ex.uid] for ex in qa.explanation_gold
    ]
    #print(qa.question.raw_txt)
    #for i,a in enumerate(qa.answers):
    #    print(f"   {'*' if i==0 else '-'}) : {a.raw_txt}")
    qa_display(qa,qa_ex_statements)

    # gather all used Keywords
    kw_all = set()
    kw_all.update( qa.question.keywords )
    for a in qa.answers:
        kw_all.update( a.keywords )
    for s in qa.question_statements:
        kw_all.update( s.keywords )
    for s in qa_ex_statements:
        kw_all.update( s.keywords )

    cols=(
        ['q+']
        +['  ']
        +[f'a{i:1d}' for i,_ in enumerate(qa.answers)]
        +['  ']
        +[f'b{i:1x}' for i,_ in enumerate(qa.question_statements)]
        +['  ']
        +[f'e{i:1x}' for i,_ in enumerate(qa_ex_statements)]
    )

    for i,kw in enumerate(sorted(list(kw_all))):
        if i%10==0:
            print(' '*40+' '.join( cols ))
        row=(
            [ (kw in qa.question.keywords) ]
            +[ False ]
            +[ (kw in a.keywords) for a in qa.answers ]
            +[ False ]
            +[ (kw in s.keywords) for s in qa.question_statements ]
            +[ False ]
            +[ (kw in s.keywords) for s in qa_ex_statements ]
        )
        print(f'{kw:>40s}'+' '.join( f'{" Y" if t else " ."}' for t in row ))

def qa_display_relatedness(qa:QuestionAnswer, statements:List[Statement], 
                            blank_diagonal=True, blank_lowertri=True)->None:
    # Get GUID -> statement dict
    guid_to_statement={s.uid:s for s in statements}
    # Look up the statements for the qa's gold explanation
    qa_ex_statements=[ 
        guid_to_statement[ex.uid] for ex in qa.explanation_gold
    ]
    qa_display(qa,qa_ex_statements)

    hdrs=(
        ['q+']
        +['  ']
        +[f'a{i:1d}' for i,_ in enumerate(qa.answers)]
        +['  ']
        +[f'b{i:1x}' for i,_ in enumerate(qa.question_statements)]
        +['  ']
        +[f'e{i:1x}' for i,_ in enumerate(qa_ex_statements)]
    )
    # Get all the keywords for each of these cols into one array
    cols=(
        [ qa.question.keywords ]
        +[set()]
        +[a.keywords for a in qa.answers]
        +[set()]
        +[s.keywords for s in qa.question_statements]
        +[set()]
        +[s.keywords for s in qa_ex_statements]
    )

    print('\n'+' '*5+' '.join( hdrs )+" : self, ext")
    blank='  '
    for row_i,row_kw in enumerate(cols):
        row_out, slf, ext=[],0,0
        for col_j,col_kw in enumerate(cols):
            c = len(row_kw & col_kw)
            s=blank if c==0 else f'{c:2d}'
            if blank_diagonal and row_i==col_j: s=blank
            if blank_lowertri and row_i>col_j:  s=blank
            row_out.append(s)
            if row_i==col_j:
                slf+=c
            else:
                ext+=c
        print(f"{hdrs[row_i]:>3s} :"+' '.join(row_out)+f" :  {slf:3d}, {ext:3d}")

def silent_average_precision_score(gold: List[str], pred: List[str]) -> float:
    if not gold or not pred:  return 0.
    correct, ap, true = 0, 0., set(gold)
    for rank, element in enumerate(pred):
        if element in true:
            correct += 1
            ap += correct / (rank + 1.)
            true.remove(element)
    return ap / len(gold)

def run_average_precision_sanity_check():
    sys.path.append("../tg2020task")
    import evaluate
    preds=[
        "abcdefghij",
        "1abcdefghij","12abcdefghij", "123abcdefghij", "1234abcdefghij", "12345abcdefghij", "123456abcdefghij", "1234567abcdefghij", "12345678abcdefghij", "123456789abcdefghij", "1234567890abcdefghij",
        "1a2b3c4d5e6f7g8h9i0j", "a2b3c4d5e6f7g8h9i0j1",
        "abcdefghi","abcdefgh","abcdefg","abcdef",
    ]
    for pred in preds:
            score = evaluate.average_precision_score(list(preds[0]), list(pred) )
            score_silent = silent_average_precision_score(list(preds[0]), list(pred) )
            print(f"{pred:>22s} {score:.4f} {score_silent:.4f}")

def analyse_outliers(limit:int, statements:List[Statement], qanda:List[QuestionAnswer], predictions_file:str):
    preds=dict() # qa_id -> [statements in order]
    with open(predictions_file, 'rt') as f:
        for l in f.readlines():
            qid, uid = l.strip().split('\t')
            if qid not in preds: preds[qid]=[]
            preds[qid].append(uid)

    # Check with competition scorer results
    #with open("/tmp/scorer/per_q.json", "rt") as f:
    #    qid2score = json.load(f)
    #print(len(qid2score), len(preds), len(qanda)); return # 410 496 496

    def evaluate_example(qa):
        #return True
        # Only keep questions where flags is in 'success' or 'ready'
        return qa.flags in "success|ready".split('|')

    statement_from_uid = { s.uid:s for s in statements }
    def statement_desc(s, reason):
        meta=''
        #if 'that something' in s.raw_txt or 'that object' in s.raw_txt:
        if 'that' in s.raw_txt:
            meta='*'
            #print(s.raw_txt)
        #if s.table=='KINDOF':
        return f"{meta}{s.table}.{reason}"

    reasons='CENTRAL|GROUNDING|LEXGLUE|ROLE|BACKGROUND|NE|NEG'.split('|')
    reason_cnt = { reason:[0,0] for reason in reasons}  # (hit,miss)

    reason_to_table_cnt={ reason:dict() for reason in reasons} # dict()=(table->cnt)
    tables=set()

    # Now got through the questions, and chart out the 'hits', 
    #   And analyse the 'misses'
    cnt, sc_tot, sc_trunc_tot, sc_max_tot=0, 0.,0.,0.
    for qa_i, qa in enumerate(qanda):
        if not evaluate_example(qa): continue
        cnt+=1
        gold=set(e.uid for e in qa.explanation_gold)
        p=preds[qa.question_id]

        uid_to_reason={ e.uid:e.reason for e in qa.explanation_gold }
        for e in qa.explanation_gold:
            t=statement_from_uid[e.uid].table
            tables.add(t)
            counts=reason_to_table_cnt[e.reason]
            if not t in counts: counts[t]=0
            counts[t]+=1

        score = silent_average_precision_score(list(gold), p)
        sc_tot+=score
        score_trunc = silent_average_precision_score(list(gold), p[:limit])
        sc_trunc_tot+=score_trunc

        hits, found, remaining=[], [], gold.copy()
        for i in range(limit):
            if i<len(p):
                uid = p[i]
                if uid in gold:
                    hits.append('X')
                    remaining.remove(uid)
                    found.append(uid)
                else:
                    hits.append('.')
            else:
                hits.append('-')
        miss=[]
        for j in remaining:
            reason = uid_to_reason[j]
            try:
                miss.append( (p.index(j), reason)  )
            except:
                miss.append( (-1, reason) ) # Missing in predictions
        miss=sorted(miss)
        misses = ','.join(f"{m[0]:d}" for m in miss)

        #score_max = silent_average_precision_score(list(gold), list(gold)[:])
        score_max = ( len(gold)-len(remaining) )/len(gold)
        sc_max_tot += score_max

        print(f"{score:.4f} {score_trunc:.4f} {score_max:.4f} : {''.join(hits)} : {misses}")
        #if score!=qid2score.get(qa.question_id.lower(), -1):
        #    print(f"{qid2score.get(qa.question_id.lower(), -1):.4f} - {qa.question_id:s}")

        s_table=['[']
        for uid in found:  # in order they were found
            reason = uid_to_reason[uid]
            s_table.append( statement_desc( statement_from_uid[uid], reason ) )
            reason_cnt[reason][0]+=1

        s_table.append(']')
        for j,reason in miss:   # in order they were missed
            if j<0:
                s_table.append( f"missing.{reason}" )    
            else:
                s_table.append( statement_desc( statement_from_uid[p[j]], reason ) )
            reason_cnt[reason][1]+=1

        print(f"   #{qa_i:4d} = {qa.question_id} :: "+' '.join(s_table))
    print(f"{sc_tot/cnt:.4f} {sc_trunc_tot/cnt:.4f} {sc_max_tot/cnt:.4f}:max (limit={limit:d})")
    
    for reason in reasons:
        hit, miss = reason_cnt[reason]
        tot=hit+miss
        print(f"{reason:>12s}({tot:4d}): {hit/tot*100.:.2f}% hit : {miss/tot*100.:.2f}% miss")

    print(f"{' '*34} "+''.join(f"{reason[:5]:>8s}" for reason in reasons))
    for t in sorted(list(tables)):
        tot = sum(reason_to_table_cnt[reason].get(t,0) for reason in reasons)
        print(f"{t:>30s} {tot:4d}"+''.join(f"{reason_to_table_cnt[reason].get(t, 0)/tot*100.:8.2f}" for reason in reasons))
            

if '__main__' == __name__:
    if False:
        run_average_precision_sanity_check()
        #exit(0)

    statements = load_statements()#regenerate=True)
    #print(len(statements))  # 13K in total (includes COMBOs)
    #print(statements[123])

    keyword_counts = get_keyword_counts_from_statements(statements)
    if False: # Let's look at the unique Keywords
        print_keyword_counts(keyword_counts)
    if False: # Let's look at the longest compound keywords
        for k,v in keyword_counts.items():
            if '_' in k:
                n_words=len(k.split('_'))
                if n_words>2:
                    print(n_words,k,v)
    if False: # Let's look at the words within compound keywords
        keyword_base=dict()
        for k,v in keyword_counts.items():
            for kw in k.split('_'):
                if kw not in keyword_base:
                    keyword_base[kw]=[]  # keep a list, so we can measure frequency
                keyword_base[kw].append(k)
        for k,kw_list in sorted(keyword_base.items()):
            print(f"{k:<20s}:{len(kw_list):4d}:"+
              ','.join(sorted(list(set(kw_list)))))

    # Parsing QuestionAnswers requires:
    #   ?? keywords (via keyword_counts) : for simple string matching

    #qanda_train = load_qanda('train')#, regenerate=True) # 1.8MB
    qanda_dev   = load_qanda('dev')#, regenerate=True)   # 400k in 496 lines
    #qanda_test  = load_qanda('test')#, regenerate=True)  # 800k
    
    if False:
        for i in [411]:  # Good examples : 
            qa_display_keywords(qanda_dev[i], statements=statements)
            qa_display_relatedness(qanda_dev[i], statements=statements)

    analyse_outliers(64, statements, qanda_dev, '../predictions/predict.dev.baseline-retrieval.txt')
    #analyse_outliers(64, statements, qanda_dev, '../predictions/predict.dev.baseline-retrieval_plus-tables.txt')
    #analyse_outliers(64, statements, qanda_dev, '../predictions/predict.dev.graph-all-to-all.txt')
    #analyse_outliers(64, statements, qanda_dev, '../predictions/predict.dev.graph-q-to-a.txt')


    """
    # TODO:
    DONE : Noun phrases into compound words =Keywords (~works)
    DONE : Separate commas out in lists in cells (works)
    DONE : Think about the word 'and' in lists in cells (works)
    DONE : Force extraction of at least 1 keyword from a token span (input flag)
    DONE : Look for badly hyphenated word '-' and fix  (exceptions file into repo)
    DONE : Look for badly transferred keywords ({'red_light'} should be {'red', 'light'}) and fix (exceptions file into repo)
    More fix-ups for keyword relabelling (ongoing)

    DONE : Check on single-word lemmatization fall-back (particularly for questions) - what POS are they (for example)
    DONE : Fix discrepancy between evaluate.py aggregate score and here

    DONE : Read Q&A datasets (extract questions, and sort answers - correct is [0])
    DONE : Do Keywords preproc on Q&A datasets

    DONE : Table of Keyword counts for QA :: KW[i] : in_q : in_a : in_ex : ...
    DONE : Matrix of Keyword overlap for QA :: (q,a,ex1,ex2,...,exN)**2

    DONE : Discover that compound keywords perform worse for TF-IDF : backtrack on that idea (move to extract_keywords_complex)
    DONE : See whether Keywords need more relabelling, etc ... 

    DONE : Ken's recall stats tell us where outlier statements fall : What are they?
    DONE : Hypothesis : Outlier statements are meta statements :: NOT PROVEN
    NOPE : Write meta-statement classifier, and add a tag/market to add as a keyword

    Have a look at how to do graph linkage directly


    Create Q&A dataset fancier preproc : MoveStatmentsFromQuestion
    Create Q&A dataset fancier preproc : MoveAssumptionsFromQuestionToAnswer

    Create Q&A dataset fancier preproc : ResolveExplanationsToSpecifics

    qa_raw = QuestionAnswer()
    qa_enh = qa_raw
    qa_enh = MoveStatmentsFromQuestion(qa_enh)
    qa_enh = MoveAssumptionsFromQuestionToAnswer(qa_enh)
    qa_enh = ResolveExplanationsToSpecifics(qa_enh)

    Make basic graph from Keyword interconnects of Statements
    Find shortest path from Q->A via https://en.wikipedia.org/wiki/Dijkstra's_algorithm


    """
