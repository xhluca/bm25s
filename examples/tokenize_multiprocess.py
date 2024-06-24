"""
# Tokenize with multiprocessing

In this example, we see how to tokenize the NQ dataset using multiprocessing.Pool
to parallelize the tokenization process. Note that this does not show how to use 
the snowball stemmer as the c object is not picklable (this should be fixable in a 
future PR) and returns strings instead of IDs/vocab, since each process cannot communicate
with the other processes to use the same dictionary.

Note that we can observe a speedup, but the per-core efficiency will go down as you use more
cores. For examples, on NQ, we observe the following:

Single Process: 110.0863s (24357.87/s)                                           
Multiprocess (4x): 61.4338s (43648.09/s)

As you can see, the time taken went down by 50s but uses 4x more threads.
"""
import multiprocessing as mp

import beir.util
from beir.datasets.data_loader import GenericDataLoader
import bm25s
from bm25s.utils.benchmark import Timer
from bm25s.utils.beir import BASE_URL

def tokenize_fn(texts):
    return bm25s.tokenize(texts=texts, return_ids=False, show_progress=False)

def chunk(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]

def unchunk(lsts):
    # merge all lsts into one list
    return [item for lst in lsts for item in lst]

if __name__ == "__main__":
    dataset = "nq"
    save_dir = "datasets"
    split = "test"
    num_processes = 4

    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)
    corpus, _, __ = GenericDataLoader(data_folder=data_path).load(split=split)

    corpus_ids, corpus_lst = [], []
    for key, val in corpus.items():
        corpus_ids.append(key)
        corpus_lst.append(val["title"] + " " + val["text"])

    del corpus

    timer = Timer("[Tokenization]")

    # let's try single process
    t = timer.start("single-threaded")
    tokens = bm25s.tokenize(texts=corpus_lst, return_ids=False)
    timer.stop(t, show=True, n_total=len(corpus_lst))

    # we will use the tokenizer class here
    corpus_chunks = chunk(corpus_lst, 1000)
    t = timer.start(f"num_processes={num_processes}")
    with mp.Pool(processes=num_processes) as pool:
        tokens_lst_chunks = pool.map(tokenize_fn, corpus_chunks)
    timer.stop(t, show=True, n_total=len(corpus_lst))
    
    tokens_lst_final = unchunk(tokens_lst_chunks)
    assert tokens == tokens_lst_final
