import re

def getSingleMAP(gold, pred):
    ranks, rank, already_output = [], 0, set()
    for ex_uid_long in pred:
        ex_uid = re.sub(r'_\d+$', '', ex_uid_long) # Clear of trailing _dd (if any)
        if not ex_uid in already_output:
            rank += 1 # This is adjusted automatically to dedupe the ggg_x combos
            already_output.add(ex_uid)
            if ex_uid in gold:
                ranks.append(rank)
    total = 0.
    for i, rank in enumerate(ranks):
        total += float(i+1)/float(rank)
    
    if len(gold)==0: return 0.
    return total / len(gold)
