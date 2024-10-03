"""
# Example: Indexing Natural Questions

This shows how to build an index of the natural questions dataset using BM25S.

To run this example, you need to install the following dependencies:

```bash
pip install bm25s[core]
```

Then, run with:

```bash
python examples/index_nq.py
```
"""

import bm25s
import Stemmer


def main(save_dir="datasets", index_dir="bm25s_indices/nq", dataset="nq"):
    print("Downloading the dataset...")
    bm25s.utils.beir.download_dataset(dataset, save_dir=save_dir)
    print("Loading the corpus...")
    corpus = bm25s.utils.beir.load_corpus(dataset, save_dir=save_dir)
    corpus_records = [
        {"id": k, "title": v["title"], "text": v["text"]} for k, v in corpus.items()
    ]
    corpus_lst = [r["title"] + " " + r["text"] for r in corpus_records]

    stemmer = Stemmer.Stemmer("english")
    corpus_tokenized = bm25s.tokenize(corpus_lst, stemmer=stemmer)

    retriever = bm25s.BM25(corpus=corpus_records)
    retriever.index(corpus_tokenized)
    print("Created BM25 index.")
    retriever.save(index_dir)
    print(f"Saved the index to {index_dir}.")
    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Peak memory usage: {mem_use:.2f} GB")


if __name__ == "__main__":
    main()
