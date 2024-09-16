"""
# Example: Use Numba to speed up the retrieval process

```bash
pip install "bm25s[full]" numba
```

To build an index, please refer to the `examples/index_and_upload_to_hf.py` script.

Now, to run this script, execute:
```bash
python examples/retrieve_with_numba.py
```
"""
import os
import Stemmer

import bm25s.hf
import bm25s

def main(dataset='scifact', dataset_dir='./datasets'):
    queries = [
        "Is chemotherapy effective for treating cancer?",
        "Is Cardiac injury is common in critical cases of COVID-19?",
    ]

    bm25s.utils.beir.download_dataset(dataset=dataset, save_dir=dataset_dir)
    corpus: dict = bm25s.utils.beir.load_corpus(dataset=dataset, save_dir=dataset_dir)
    corpus_records = [
        {'id': k, 'title': v["title"], 'text': v["text"]} for k, v in corpus.items()
    ]
    corpus_lst = [r["title"] + " " + r["text"] for r in corpus_records]

    retriever = bm25s.BM25(corpus=corpus_records, backend='numba')
    retriever.index(corpus_lst)
    # corpus=corpus_records is optional, only used when you are calling retrieve and want to return the documents

    # Tokenize the queries
    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
    queries_tokenized = tokenizer.tokenize(queries)
    # Retrieve the top-k results
    results = retriever.retrieve(queries_tokenized, k=3)
    # show first results
    result = results.documents[0]
    print(f"First score (# 1 result): {results.scores[0, 0]:.4f}")
    print(f"First result id (# 1 result): {result[0]['id']}")
    print(f"First result title (# 1 result): {result[0]['title']}")

if __name__ == "__main__":
    main()