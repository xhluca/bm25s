"""
# Example: Retrieve from pre-built index of Natural Questions

This shows how to load an index built with BM25.index and saved with BM25.save, and retrieve
the top-k results for a set of queries from the Natural Questions dataset, via BEIR library.

To run this example, you need to install the following dependencies:

```bash
pip install beir bm25s[full]
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
import beir.util
from beir.datasets.data_loader import GenericDataLoader
import Stemmer

import bm25s
from bm25s.utils.beir import BASE_URL

def main(index_dir="bm25s_indices/nq", data_dir="datasets", dataset="nq", mmap=True):
    if mmap:
        print("Using memory-mapped index (mmap) to reduce memory usage.")
    
    # Load the queries from BEIR
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), data_dir)
    loader = GenericDataLoader(data_folder=data_path)
    loader._load_queries()
    queries_lst = list(loader.queries.values())[:1000]

    # Tokenize the queries
    stemmer = Stemmer.Stemmer("english")
    queries_tokenized = bm25s.tokenize(queries_lst, stemmer=stemmer)

    # Load the BM25 index and retrieve the top-k results
    retriever = bm25s.BM25.load(index_dir, mmap=mmap, load_corpus=True)
    results = retriever.retrieve(queries_tokenized, k=20)
    
    first_result = results.documents[0]
    print(f"First score (# 1 result):{results.scores[0, 0]}")
    print(f"First result (# 1 result):\n{first_result[0]}")

if __name__ == "__main__":
    main()