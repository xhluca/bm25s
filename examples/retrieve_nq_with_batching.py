"""
# Example: Retrieve from pre-built index of Natural Questions

This is a modified version of the `examples/retrieve_nq.py` script that uses batching to
even reduce memory usage further. This script loads the queries in batches and retrieves
the top-k results for each batch, clearing the memory after each batch.

To run this example, you need to install the following dependencies:

```bash
pip install bm25s[core]
```

To build an index, please refer to the `examples/index_nq.py` script. You
can run this script with:

```bash
python examples/index_nq.py
```

Then, run this script with:

```bash
python examples/retrieve_nq.py
```
"""

import bm25s
import Stemmer
from tqdm import tqdm


def main(index_dir="bm25s_indices/nq", data_dir="datasets", dataset="nq", bsize=20):
    mmap = True
    print("Using memory-mapped index (mmap) to reduce memory usage.")

    timer = bm25s.utils.benchmark.Timer("[BM25S]")

    queries = bm25s.utils.beir.load_queries(dataset, save_dir=data_dir)
    qrels = bm25s.utils.beir.load_qrels(dataset, split="test", save_dir=data_dir)
    queries_lst = [v["text"] for k, v in queries.items() if k in qrels]
    print(f"Loaded {len(queries_lst)} queries.")

    # Tokenize the queries
    stemmer = Stemmer.Stemmer("english")
    queries_tokenized = bm25s.tokenize(queries_lst, stemmer=stemmer, return_ids=False)

    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Initial memory usage: {mem_use:.2f} GB")

    # Load the BM25 index and retrieve the top-k results
    print("Loading the BM25 index...")
    t = timer.start("Loading index")
    retriever = bm25s.BM25.load(index_dir, mmap=mmap, load_corpus=True)
    retriever.backend = "numba"
    num_docs = retriever.scores["num_docs"]
    timer.stop(t, show=True, n_total=num_docs)

    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Memory usage after loading the index: {mem_use:.2f} GB")

    print("Retrieving the top-k results...")
    t = timer.start("Retrieving")

    batches = []

    for i in tqdm(range(0, len(queries_lst), bsize)):
        batches.append(retriever.retrieve(queries_tokenized[i : i + bsize], k=10))
        
        # reload the corpus and scores to free up memory
        retriever.load_scores(save_dir=index_dir, mmap=mmap, num_docs=num_docs)
        if isinstance(retriever.corpus, bm25s.utils.corpus.JsonlCorpus):
            retriever.corpus.load()

    results = bm25s.Results.merge(batches)

    timer.stop(t, show=True, n_total=len(queries_lst))

    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Final (peak) memory usage: {mem_use:.2f} GB")

    print("-" * 50)
    first_result = results.documents[0]
    print(f"First score (# 1 result): {results.scores[0, 0]:.4f}")
    print(f"First result (# 1 result):\n{first_result[0]}")


if __name__ == "__main__":
    main()
