"""
# Example: Indexing Natural Questions

This shows how to build an index of the natural questions dataset using BM25S.

To run this example, you need to install the following dependencies:

```bash
pip install beir bm25s PyStemmer
```

Then, run with:

```bash
python examples/index_nq.py
```
"""
import beir.util
from beir.datasets.data_loader import GenericDataLoader
import Stemmer  # from PyStemmer

import bm25s
from bm25s.utils.beir import BASE_URL


def main(save_dir="datasets", index_dir="bm25s_indices/nq", dataset="nq"):
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)
    corpus, _, __ = GenericDataLoader(data_folder=data_path).load(split="test")
    corpus_records = [
        {'id': k, 'title': v["title"], 'text': v["text"]} for k, v in corpus.items()
    ]
    corpus_lst = [r["title"] + " " + r["text"] for r in corpus_records]

    stemmer = Stemmer.Stemmer("english")
    corpus_tokenized = bm25s.tokenize(corpus_lst, stemmer=stemmer)

    retriever = bm25s.BM25(corpus=corpus_records)
    retriever.index(corpus_tokenized)
    retriever.save(index_dir)

if __name__ == "__main__":
    main()