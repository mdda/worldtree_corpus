import json
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from fire import Fire
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from tqdm import tqdm

from dataset import Statement, QuestionAnswer, load_qanda, load_statements
from extra_data import SplitEnum, analyze_lengths
from losses import APLoss
from preprocessors import (
    TextProcessor,
    SpacyProcessor,
    SentenceSplitProcessor,
    KeywordProcessor,
    OnlyGoldWordsProcessor,
)
from rankers import Ranker, StageRanker, IterativeRanker, deduplicate
from vectorizers import BM25Vectorizer, TruncatedSVDVectorizer

#sys.path.append("../tg2020task")
#sys.path.append("../tg2021task")
#import evaluate


class Data(BaseModel):
    root: str = "../data"

    #root_gold: str = "../tg2020task"
    root_gold: str = "../tg2021task/data-evalperiod"
    ## During hyperparam optimisation
    #root_gold: str = "/mnt/rdai/reddragon/research/textgraphs/worldtree_corpus/tg2021task/data-evalperiod"

    data_split: SplitEnum = SplitEnum.dev
    statements: Optional[List[Statement]]
    questions: Optional[List[QuestionAnswer]]
    uid_to_statements: Optional[Dict[str, List[Statement]]]

    @property
    def path_gold(self) -> Path:
        return Path(self.root_gold) / f"questions.{self.data_split}.tsv"

    @property
    def json_gold(self) -> Path:
        return Path(self.root_gold) / f"wt-expert-ratings.{self.data_split}.json"

    @staticmethod
    def load_jsonl(path: Path) -> List[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                assert line
                records.append(json.loads(line))
        return records

    def load(self):
        self.statements = load_statements()
        self.questions = load_qanda(self.data_split)

        self.uid_to_statements = {}
        for s in self.statements:
            self.uid_to_statements.setdefault(s.uid_base, []).append(s)

    def analyze(self):
        num_explains = [len(q.explanation_gold) for q in self.questions]
        info = dict(
            statements=len(self.statements),
            questions=len(self.questions),
            num_explains=analyze_lengths(num_explains),
        )
        if self.data_split != SplitEnum.test:
            len_explains: List[int] = []
            for q in self.questions:
                for e in q.explanation_gold:
                    statements = self.uid_to_statements.get(e.uid)
                    if statements is None:
                        print(dict(missing_uid=e.uid))
                    else:
                        state = self.uid_to_statements[e.uid][0]
                        len_explains.append(len(state.raw_txt))
            len_qns = [len(q.question.raw_txt) for q in self.questions]
            info.update(
                len_explains=analyze_lengths(len_explains),
                len_qns=analyze_lengths(len_qns),
            )
        print(info)


class Prediction(BaseModel):
    qid: str
    uids: List[str]


class Retriever(BaseModel):
    preproc: TextProcessor = TextProcessor()
    vectorizer: TfidfVectorizer = BM25Vectorizer()
    ranker: Ranker = Ranker()
    limit: int = -1

    class Config:
        arbitrary_types_allowed = True

    def make_pred(self, i_query: int, rank: List[int], data: Data) -> Prediction:
        uids = [data.statements[i].uid_base for i in rank]
        uids = deduplicate(uids, limit=self.limit)
        return Prediction(qid=data.questions[i_query].question_id, uids=uids)

    def make_query(self, q: QuestionAnswer, data: Data) -> str:
        assert data
        return self.preproc.run(q.question) + " " + self.preproc.run(q.answers[0])

    def rank(self, queries: List[str], statements: List[str]) -> np.ndarray:
        self.vectorizer.fit(statements + queries)
        return self.ranker.run(
            self.vectorizer.transform(queries), self.vectorizer.transform(statements)
        )

    def run(self, data: Data) -> List[Prediction]:
        statements: List[str] = [self.preproc.run(s) for s in data.statements]
        queries: List[str] = [self.make_query(q, data) for q in data.questions]
        ranking = self.rank(queries, statements)
        preds: List[Prediction] = []
        for i in tqdm(range(len(ranking)), desc="Retriever"):
            preds.append(self.make_pred(i, list(ranking[i]), data))
        return preds


class PredictManager(BaseModel):
    file_pattern: str
    fold_marker: str = "FOLD"
    sep_line: str = "\n"
    sep_field: str = "\t"

    def make_path(self, data_split: str) -> Path:
        assert self.file_pattern.count(self.fold_marker) == 1
        path = Path(self.file_pattern.replace(self.fold_marker, data_split))
        path.parent.mkdir(exist_ok=True)
        return path

    def write(self, preds: List[Prediction], data_split: str, limit=None):
        if limit is None: limit=9999999 # Bigger than the dataset...
        lines = []
        for p in preds:
            for u in p.uids[:limit]:
                lines.append(self.sep_field.join([p.qid, u]))

        path = self.make_path(data_split)
        with open(path, "w") as f:
            f.write(self.sep_line.join(lines))
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(preds, f)

    def read(self, data_split: str) -> List[Prediction]:
        start = time.time()
        qid_to_uids = {}
        path = self.make_path(data_split)
        with open(path) as f:
            for line in f:
                line = line.strip()
                qid, uid = line.split(self.sep_field)
                qid_to_uids.setdefault(qid, []).append(uid)
        preds = [Prediction(qid=qid, uids=uids) for qid, uids in qid_to_uids.items()]
        duration = time.time() - start
        print(dict(read=path, duration=round(duration, 3)))
        return preds

    def read_pickle(self, data_split: str) -> List[Prediction]:
        # About 50x faster than reading .txt, with 7x smaller file
        start = time.time()
        path = self.make_path(data_split).with_suffix(".pkl")
        with open(path, "rb") as f:
            preds = pickle.load(f)

        duration = time.time() - start
        print(dict(read=path, duration=round(duration, 3)))
        return preds

import evaluate2020  # Just copied over locally here...
class Scorer2020(BaseModel):
    @staticmethod
    def run(path_gold: Path, path_predict: Path) -> Dict[str, float]:
        gold = evaluate2020.load_gold(str(path_gold))  # noqa
        pred = evaluate2020.load_pred(str(path_predict))  # noqa
        qid2score = {}

        def _callback(qid, score):
            qid2score[qid] = score

        score_fn = evaluate2020.mean_average_precision_score  # noqa
        mean_ap = score_fn(gold, pred, callback=_callback)
        print(dict(mean_ap=mean_ap))
        with open("/tmp/per_q.json", "wt") as f:
            json.dump(qid2score, f)
        return qid2score


class ResultAnalyzer(BaseModel):
    thresholds: List[int] = [64, 100, 128, 256, 512]
    loss_fn: nn.Module = APLoss()

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def filter_qns(
        data: Data, preds: List[Prediction]
    ) -> Tuple[List[QuestionAnswer], List[Prediction]]:
        assert len(data.questions) == len(preds)
        pairs = []
        for q, p in zip(data.questions, preds):
            if q.explanation_gold:
                pairs.append((q, p))
            else:
                print(dict(qn_no_explains=q.question_id))

        qns, preds = zip(*pairs)
        return qns, preds

    def run_threshold(
        self, data: Data, preds: List[Prediction], threshold: int
    ) -> float:
        qns, preds = self.filter_qns(data, preds)
        scores = []

        for q, p in zip(qns, preds):
            assert q.question_id == p.qid
            predicted = set(p.uids[:threshold])
            gold = set([e.uid for e in q.explanation_gold])
            if gold:
                true_pos = predicted.intersection(gold)
                recall = len(true_pos) / len(gold)
                scores.append(recall)

        return sum(scores) / len(scores)

    def run_map_loss(
        self, data: Data, preds: List[Prediction], top_n: int = None
    ) -> float:
        if top_n is None:
            top_n = len(preds[0].uids)
        else:
            top_n = min(len(preds[0].uids), top_n)

        qns, preds = self.filter_qns(data, preds)
        losses = []
        scores = torch.div(1.0, torch.arange(top_n) + 1)  # [1, 0.5, 0.3  ... 0]
        scores = torch.unsqueeze(scores, dim=0)

        for i, q in enumerate(qns):
            uids = preds[i].uids[:top_n]
            uids_gold = set([e.uid for e in q.explanation_gold])
            labels = [int(u in uids_gold) for u in uids]
            if sum(labels) == 0:
                continue
            loss = self.loss_fn(scores, torch.Tensor(labels).unsqueeze(dim=0))
            losses.append(loss.item())
        return sum(losses) / len(losses)

    @staticmethod
    def count_qns_no_hits(data: Data, preds: List[Prediction], top_n: int) -> int:
        count = 0
        for q, p in zip(data.questions, preds):
            uids_gold = set([e.uid for e in q.explanation_gold])
            if not uids_gold.intersection(p.uids[:top_n]):
                count += 1
        return count

    def run(self, data: Data, preds: List[Prediction]):
        records = []
        for threshold in self.thresholds + [len(preds[0].uids)]:
            recall = self.run_threshold(data, preds, threshold)
            qns_no_hits = self.count_qns_no_hits(data, preds, threshold)
            map_loss = self.run_map_loss(data, preds, threshold)
            records.append(
                dict(
                    threshold=threshold,
                    recall=recall,
                    qns_no_hits=qns_no_hits,
                    map_loss=map_loss,
                )
            )
        df = pd.DataFrame(records)
        print(df)


def main(
    data_split=SplitEnum.dev,
    output_pattern="../predictions/predict.FOLD.baseline-retrieval.txt",
):
    data = Data(data_split=data_split)
    data.load()
    data.analyze()
    manager = PredictManager(file_pattern=output_pattern)

    retrievers = [
        Retriever(),  # Dev MAP=0.3965, recall@512=0.7911
        Retriever(preproc=SpacyProcessor()),  # Dev MAP=0.4587, recall@512=0.8849
        Retriever(
            preproc=SpacyProcessor(remove_stopwords=True)
        ),  # Dev MAP=0.4615, recall@512=0.8780
        Retriever(  # This shows that only the last 4 sentences are necessary...
            preproc=SentenceSplitProcessor(max_sentences=4),
        ),  # Dev MAP=0.4615, recall@512=0.8795
        Retriever(preproc=KeywordProcessor()),  # Dev MAP=0.4529, recall@512=0.8755
        Retriever(
            preproc=KeywordProcessor(), ranker=StageRanker()
        ),  # Dev MAP=0.4575, recall@512=0.9095
        # Maybe this dense vectorizer can make useful features for deep learning methods
        Retriever(
            vectorizer=TruncatedSVDVectorizer(BM25Vectorizer(), n_components=768),
            preproc=KeywordProcessor(),
        ),  # Dev MAP:  0.3741, recall@512=0.8772
        Retriever(
            preproc=KeywordProcessor(), ranker=IterativeRanker()
        ),  # Dev MAP=0.4704, recall@512=0.8910
        Retriever(
            preproc=KeywordProcessor(),
            ranker=StageRanker(num_per_stage=[16, 32, 64, 128], scale=1.5),
        ),  # Dev MAP=0.4586, recall@512=0.9242
        # Loading SpacyVectorizer has an annoying delay
        # Retriever(
        #     preproc=SpacyProcessor(remove_stopwords=True),
        #     vectorizer=SpacyVectorizer(),
        #     ranker=WordEmbedRanker(),
        # ),  # Dev MAP=0.01436, recall@512=0.4786
        # From hyperopt_retrieval.py
        Retriever(  # This shows that keyword generation models have potential
            preproc=OnlyGoldWordsProcessor(
                questions=data.questions,
                statements=data.statements,
                add_all_gold_words=True,
            ),
        ),  # Dev MAP=0.8118, recall@512=1.000
        Retriever(
            preproc=SpacyProcessor(remove_stopwords=True),
            ranker=StageRanker(num_per_stage=[1, 2, 4, 8, 16], scale=1.25),
            vectorizer=BM25Vectorizer(binary=True, use_idf=True, k1=2.0, b=0.5),
        ),  # Dev MAP=0.4861, recall@512=0.9345
    ]
    r = retrievers[-1]
    preds = r.run(data)
    manager.write(preds, data_split, limit=100)
    if data_split != SplitEnum.test:
        Scorer2020().run(data.path_gold, manager.make_path(data_split))
        ResultAnalyzer2020().run(data, preds)


import nni

def process_expert_gold(expert_preds: List) -> Dict[str, Dict[str, float]]:
    return {
        pred["qid".lower()]: {
            data["uuid"]: data["relevance"] for data in pred["documents"]
        }
        for pred in expert_preds
    }


class Scorer(BaseModel):
    #@staticmethod
    #def get_coverage(gold_explanations, score_min, 
  
    @staticmethod
    def run(path_gold: Path, preds: List[Prediction]) -> float:
        # https://colab.research.google.com/drive/1uexs4-ir0E9dbAsGPbCUJDAhmx0nwRx-?usp=sharing
        with open(path_gold, "rt") as f:
            #gold_explanations = evaluate.process_expert_gold(json.load(f)["rankingProblems"])
            gold_explanations = process_expert_gold(json.load(f)["rankingProblems"])
        
        #print(f"{path_gold} : {len(gold_explanations)}, {len(preds)}")
        score_min_arr = [0,1,2,3,4,5]
        limit_arr     = [100, 128, 200, 256, 300, 400, 512]

        coverage_tot=np.zeros( (len(score_min_arr), len(limit_arr)) )
        coverage_cnt=coverage_tot.copy()
        
        for i, score_min in enumerate(score_min_arr):
          for p in preds:
            k = p.qid
            
            # Get list of keys in expert ranking in sorted order (if they're worth >0)
            #expert_order = sorted([(v,k) for k,v in gold_explanations[k].items() if v>0.], reverse=True)
            expert_set = set(id for id,v in gold_explanations[k].items() if v>score_min) 
            
            if len(expert_set)==0: continue

            for j, limit in enumerate(limit_arr):
              predicted_uids = p.uids[:limit]
              coverage_ids=[ u for u in predicted_uids if u in expert_set ]
              
              #print(expert_set)
              #print([u for u in predicted_uids if u not in expert_set])
              
              #print(f"Want {len(expert_set)} and found {len(coverage_ids)} in first {len(predicted_uids)} predictions")
              #coverage_arr.append( len(coverage_ids)/len(expert_set) )
              coverage_tot[i,j] += len(coverage_ids)/len(expert_set)
              coverage_cnt[i,j] += 1
              
          #coverage = sum(coverage_arr)/len(coverage_arr)
          #print(f"{score_min:.1f} : {coverage:.4f}")
          #coverage_by_score[score_min]=coverage
        
        coverage_av = coverage_tot/coverage_cnt
        for i,score_min in enumerate(score_min_arr):
          print( f"{score_min:4d} : "+(', '.join([f"{coverage_av[i,j]:.4f}" for j,limit in enumerate(limit_arr)])) )
            
        #return (coverage_av[:,0]).mean()  # This is the average score for limit=100
        return (coverage_av[:,2]).mean()  # This is the average score for limit=200


def hyperopt(
    #data_split=SplitEnum.train,
    data_split=SplitEnum.dev,
    #data_split=SplitEnum.test,
    output_pattern="../predictions/predict.FOLD.baseline-retrieval.hyperopt.txt",
):
    data = Data(data_split=data_split)
    data.load()
    data.analyze()
    
    #print(data.json_gold)
    #exit(0)
    
    manager = PredictManager(file_pattern=output_pattern)

    """@XXnni.variable(nni.choice(False, True), name=remove_stopwords)"""  # proven by top 20% of scores
    remove_stopwords=True
  
    """@XXnni.variable(nni.uniform(1, 4), name=nps_base)""" 
    """@XXnni.variable(nni.uniform(1, 2.1), name=nps_base)""" 
    """@nni.variable(nni.uniform(1.0, 1.2), name=nps_base)"""  # On training set
    nps_base=1.0
  
    """@XXnni.variable(nni.uniform(1, 3), name=nps_ratio)"""
    """@XXnni.variable(nni.uniform(2, 3), name=nps_ratio)"""
    """@nni.variable(nni.uniform(2.4, 2.9), name=nps_ratio)"""  # On training set
    nps_ratio=2.7
  
    nps, nps_arr=nps_base, []
    for _ in range(5):
      nps_arr.append( int(nps) )
      nps*=nps_ratio
    
    """@XXnni.variable(nni.uniform(1.1, 1.9), name=scale)"""  # Initial range
    """@XXnni.variable(nni.uniform(0.7, 1.4), name=scale)"""
    """@nni.variable(nni.uniform(1.1, 1.2), name=scale)"""  # On training set
    scale=1.15
    
    ## https://www.elastic.co/blog/practical-bm25-part-3-considerations-for-picking-b-and-k1-in-elasticsearch
    """@XXnni.variable(nni.choice(False, True), name=bm25_binary)"""
    bm25_binary=False  # Changed from original 'True' based on training set
    
    """@XXnni.variable(nni.choice(False, True), name=use_idf)""" # False proven to be a bad choice
    use_idf=True
    
    # The default values of b = 0.75 and k1 = 1.2 work pretty well for most corpuses, so you’re likely fine with the defaults. 
    #   ... seem to show the optimal b to be in a range of 0.3-0.9
    #   ... show the optimal k1 to be in a range of 0.5-2.0  [k1 is typically evaluated in the 0 to 3 range]
    """@XXnni.variable(nni.uniform(0.3, 0.9), name=b)"""
    """@XXnni.variable(nni.uniform(0.3, 0.7), name=b)"""
    """@nni.variable(nni.uniform(0.5, 0.7), name=b)"""    # On training set
    b=0.6
    """@XXnni.variable(nni.uniform(0.5, 2.5), name=k1)"""
    """@XXnni.variable(nni.uniform(0.5, 2.0), name=k1)"""
    """@nni.variable(nni.uniform(0.5, 0.75), name=k1)"""   # On training set
    k1=0.625  # Changed from original 2.0  based on training set

    r=  Retriever(
            preproc=SpacyProcessor(remove_stopwords=remove_stopwords),
            ranker=StageRanker(num_per_stage=nps_arr, scale=scale),   # [1, 2, 4, 8, 16]
            vectorizer=BM25Vectorizer(binary=bm25_binary, use_idf=use_idf, k1=k1, b=b),
        ) 
        
    preds = r.run(data)
    manager.write(preds, data_split, limit=200)
    if data_split != SplitEnum.test:
        #Scorer().run(data.path_gold, manager.make_path(data_split))
        #ResultAnalyzer().run(data, preds)
        coverage = Scorer().run(data.json_gold, preds)
        print(f"Coverage : {coverage:.4f}")
        """@nni.report_final_result(coverage)"""
        

if __name__ == "__main__":
    #  https://github.com/google/python-fire
    #Fire(main)
    
    # https://nni.readthedocs.io/en/stable/Tutorial/AnnotationSpec.html?highlight=nni.variable#annotate-variables
    '''@XXXnni.get_next_parameter()'''
    Fire(hyperopt)

"""
Original 'dev' retrieval coverage
0.0 : 0.6592
1.0 : 0.6826
2.0 : 0.7511
3.0 : 0.8013
4.0 : 0.9057
5.0 : 0.9101
Coverage : 0.7850

   0 : 0.6637, 0.7024, 0.7699, 0.7987, 0.8154, 0.8442, 0.8674
   1 : 0.6869, 0.7258, 0.7911, 0.8179, 0.8338, 0.8609, 0.8828
   2 : 0.7528, 0.7893, 0.8439, 0.8643, 0.8761, 0.8956, 0.9129
   3 : 0.8034, 0.8348, 0.8818, 0.8975, 0.9071, 0.9212, 0.9343
   4 : 0.9087, 0.9258, 0.9484, 0.9558, 0.9622, 0.9745, 0.9805
   5 : 0.9210, 0.9351, 0.9516, 0.9584, 0.9658, 0.9769, 0.9849
Coverage : 0.8645  (equiv to 0.7894 on first column)

Original 'train' retrieval coverage
0.0 : 0.6637
1.0 : 0.6873
2.0 : 0.7543
3.0 : 0.8018
4.0 : 0.9037
5.0 : 0.9260
Coverage : 0.7895


"""
