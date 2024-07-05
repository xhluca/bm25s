"""
Sometimes you might be interested in benchmarking BM25 on BEIR. bm25s makes this straightforward in Python.

To install:

```
pip install bm25s[core] beir
```

Now, run this script, you can modify the `run_benchmark()` part to use the datase you want to test on.
"""
import json
import os
from pathlib import Path
import time

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import numpy as np
from tqdm.auto import tqdm
import Stemmer

import bm25s
from bm25s.utils.benchmark import get_max_memory_usage, Timer
from bm25s.utils.beir import (
    BASE_URL,
    clean_results_keys,
)

def postprocess_results_for_eval(results, scores, query_ids):
    """
    Given the queried results and scores output by BM25S, postprocess them
    to be compatible with BEIR evaluation functions.
    query_ids is a list of query ids in the same order as the results.
    """

    results_record = [
        {"id": qid, "hits": results[i], "scores": list(scores[i])}
        for i, qid in enumerate(query_ids)
    ]

    result_dict_for_eval = {
        res["id"]: {
            docid: float(score) for docid, score in zip(res["hits"], res["scores"])
        }
        for res in results_record
    }

    return result_dict_for_eval

def run_benchmark(dataset, save_dir="datasets"):
    #### Download dataset and unzip the dataset
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)
    split = "test" if dataset != "msmarco" else "dev"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["title"] + " " + val["text"])
    del corpus

    qids, queries_lst = [], []
    for key, val in queries.items():
        qids.append(key)
        queries_lst.append(val)

    stemmer = Stemmer.Stemmer("english")
    
    corpus_tokens = bm25s.tokenize(
        corpus_lst, stemmer=stemmer, leave=False
    )

    del corpus_lst

    query_tokens = bm25s.tokenize(
        queries_lst, stemmer=stemmer, leave=False
    )

    model = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    model.index(corpus_tokens, leave_progress=False)
    
    ############## BENCHMARKING BEIR HERE ##############
    queried_results, queried_scores = model.retrieve(
        query_tokens, corpus=corpus_ids, k=1000, n_threads=4
    )

    results_dict = postprocess_results_for_eval(queried_results, queried_scores, qids)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results_dict, [1, 10, 100, 1000]
    )

    print(ndcg)
    print(recall)
    
    return ndcg, _map, recall, precision

if __name__ == "__main__":
  ndcg, _map, recall, precision = run_benchmark("scidocs")  # Change to dataset you want
