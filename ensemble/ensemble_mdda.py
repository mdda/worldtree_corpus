import os

from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

from fire import Fire

from dataset import QuestionRatingDataset #, ExplanationDataset, PredictDataset
from retriever import PredictManager, Prediction
from evaluate import mean_average_ndcg

import nni

print(f"{nni.get_experiment_id()=}")
running_local = (nni.get_experiment_id()=='STANDALONE')

if running_local:
  REPO_BASE='..'
else:
  REPO_BASE='/mnt/rdai/reddragon/research/textgraphs/worldtree_corpus'


def read_preds(version, fold='dev'):
  fold = fold.replace('dev', 'val')
  
  # Copy the .val. or .test. files under ./predictions/ ...
  fname = f"{REPO_BASE}/predictions/predict.{fold:s}.model_version{version:d}.txt"

  print(f"Loading {fname}")
  return PredictManager.read(fname)



def make_preds(pred_arr):
  # A list of all the relevant qids
  qids = [ p.qid for p in pred_arr[0][0] ]
  
  #pred_dict=dict()
  #for pred in pred_arr:
  
  # Let's guarantee these are in the same order...
  for i, qid in enumerate(qids):
    for j, tup in enumerate(pred_arr):
      pred=tup[0][i]
      if pred.qid != qid:
        print("Mismatching qid for prediction set {j} at {i}")
        exit(0)
    
  pred_output=[]
  for i, qid in enumerate(qids):
    tot_value_for_ex=dict()
    for j, tup in enumerate(pred_arr):  # Go through each of the things to be ensembled
      pred = tup[0][i]
      top, base = tup[1], tup[2]
      pred_len=len(pred.eids)
      for rank, eid in enumerate(pred.eids):  # And add all of their data (based on fractional rank) to the totals
        frac = rank/pred_len
        if eid not in tot_value_for_ex: tot_value_for_ex[eid]=0.
        tot_value_for_ex[eid] += (1.-frac)*top + base
    # Now, we have a dictionary of tot_value_for_ex, need to sort this into descending order of score
    eids = [ eid for eid, v in sorted(tot_value_for_ex.items(), key=lambda i: i[1], reverse=True) ]
    pred_output.append( Prediction(qid=qid, eids=eids) )
  
  #return pred_arr[0][0]
  return pred_output


def hyperopt(
  fold='dev',
  #fold='test',
  save_file=False,
  #save_file="ens.dev.txt",
  ):
    # Load in all the files
    #for version in [9,13]: # ,14,15
    p0 = read_preds(9, fold)
    p1 = read_preds(13, fold)
    p2 = read_preds(14, fold)
    p3 = read_preds(15, fold)

    """@nni.variable(nni.uniform(0.1, 4.0), name=p0_top)"""
    p0_top=2.5
    """@nni.variable(nni.uniform(0.1, 4.0), name=p1_top)"""
    p1_top=1.5
    """@nni.variable(nni.uniform(0.1, 4.0), name=p2_top)"""
    p2_top=1.0
    """@nni.variable(nni.uniform(0.1, 4.0), name=p3_top)"""
    p3_top=3.5

    """@XXnni.variable(nni.uniform(0.0, 1.0), name=p0_base)"""
    p0_base=0.0
    """@XXnni.variable(nni.uniform(0.0, 1.0), name=p1_base)"""
    p1_base=0.0
    """@XXnni.variable(nni.uniform(0.0, 1.0), name=p2_base)"""
    p2_base=0.0
    """@XXnni.variable(nni.uniform(0.0, 1.0), name=p3_base)"""
    p3_base=0.0

    pred_ens = make_preds([
      [p0, p0_top, p0_base],
      [p1, p1_top, p1_base],
      [p2, p2_top, p2_base],
      [p3, p3_top, p3_base],
    ])

    if save_file:
      # Save off the resulting dataset
      PredictManager.write(save_file, pred_ens)
      pass
    
    if fold=='dev':
      dataset = QuestionRatingDataset(f"{REPO_BASE}/data/wt-expert-ratings.dev.json")
      ge = dataset.gold_predictions
      
      ndcg = mean_average_ndcg(ge, pred_ens, 0, oracle=False)
      print(f"ensemble_mdda : nfcg = {ndcg:.4f}")
      
      """@nni.report_final_result(ndcg)"""


if __name__ == "__main__":
    #  https://github.com/google/python-fire
    #Fire(main)
    
    # https://nni.readthedocs.io/en/stable/Tutorial/AnnotationSpec.html?highlight=nni.variable#annotate-variables
    # https://nni.readthedocs.io/en/stable/Tutorial/AnnotationSpec.html

    # cd ./ensembler
    
    # cp ../dataset.py .
    # cp ../retriever.py .
    # cp ../evaluate.py .

    # Start this under 'nni' with :: 
    # nnictl create --config ensemble_mdda.yml
    
    
    
    # and clean up with:
    # nnictl stop
    
    '''@XXXnni.get_next_parameter()'''
    Fire(hyperopt)
    
"""
mv ~/Downloads/predict.val.model_version* ./predictions/
ln -s ../tg2021task/data-evalperiod/wt-expert-ratings.dev.json data/

cd tg2021task

# Each takes ~100sec on 1 core on square

./evaluate.py --gold data-evalperiod/wt-expert-ratings.dev.json ../predictions/predict.val.model_version9.txt
Mean NDCG Score : 0.7680

./evaluate.py --gold data-evalperiod/wt-expert-ratings.dev.json ../predictions/predict.val.model_version13.txt
Mean NDCG Score : 0.7574

./evaluate.py --gold data-evalperiod/wt-expert-ratings.dev.json ../predictions/predict.val.model_version14.txt
Mean NDCG Score : 0.7544

./evaluate.py --gold data-evalperiod/wt-expert-ratings.dev.json ../predictions/predict.val.model_version15.txt
Mean NDCG Score : 0.7563


"""
