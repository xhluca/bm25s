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

from pathlib import Path
import bm25s
import Stemmer


def main(save_dir="datasets", index_dir="bm25s_indices/", dataset="nq"):
    index_dir = Path(index_dir) / dataset
    index_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading the dataset...")
    bm25s.utils.beir.download_dataset(dataset, save_dir=save_dir)
    print("Loading the corpus...")
    corpus = bm25s.utils.beir.load_corpus(dataset, save_dir=save_dir)
    corpus_records = [
        {"id": k, "title": v["title"], "text": v["text"]} for k, v in corpus.items()
    ]
    corpus_lst = [r["title"] + " " + r["text"] for r in corpus_records]

    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
    corpus_tokens = tokenizer.tokenize(corpus_lst, return_as="tuple")

    retriever = bm25s.BM25(corpus=corpus_records, backend="numba")
    retriever.index(corpus_tokens)
    
    retriever.save(index_dir)
    tokenizer.save_vocab(index_dir)
    tokenizer.save_stopwords(index_dir)
    print(f"Saved the index to {index_dir}.")
    
    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Peak memory usage: {mem_use:.2f} GB")


if __name__ == "__main__":
    main(dataset='nq')
