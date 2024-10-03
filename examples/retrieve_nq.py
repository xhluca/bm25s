"""
# Example: Retrieve from pre-built index of Natural Questions

This shows how to load an index built with BM25.index and saved with BM25.save, and retrieve
the top-k results for a set of queries from the Natural Questions dataset, via BEIR library.

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

import numpy as np
import bm25s
import Stemmer
from tqdm import tqdm


def main(index_dir="bm25s_indices/nq", data_dir="datasets", dataset="nq", mmap=True):
    if mmap:
        print("Using memory-mapped index (mmap) to reduce memory usage.")

    timer = bm25s.utils.benchmark.Timer("[BM25S]")

    print("Loading the queries...")
    queries = bm25s.utils.beir.load_queries(dataset, save_dir=data_dir)
    queries_lst = [q["text"] for q in queries.values()]

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
    num_docs = retriever.scores['num_docs']
    timer.stop(t, show=True, n_total=num_docs)

    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Memory usage after loading the index: {mem_use:.2f} GB")

    print("Retrieving the top-k results...")
    t = timer.start("Retrieving")
    results = retriever.retrieve(queries_tokenized, k=10)
    timer.stop(t, show=True, n_total=len(queries_lst))

    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Final (peak) memory usage: {mem_use:.2f} GB")

    print("-" * 50)
    first_result = results.documents[0]
    print(f"First score (# 1 result): {results.scores[0, 0]:.4f}")
    print(f"First result (# 1 result):\n{first_result[0]}")


if __name__ == "__main__":
    main(mmap=True)
