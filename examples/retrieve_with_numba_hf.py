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

def main(repo_name="xhluca/bm25s-fiqa-index"):
    queries = [
        "Is chemotherapy effective for treating cancer?",
        "Is Cardiac injury is common in critical cases of COVID-19?",
    ]

    retriever = bm25s.hf.BM25HF.load_from_hub(
        repo_name, load_corpus=False, mmap=False
    )

    retriever.backend = "numba"  # this can also be set during initialization of the retriever

    # Tokenize the queries
    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
    queries_tokenized = tokenizer.tokenize(queries)

    # Retrieve the top-k results
    results = retriever.retrieve(queries_tokenized, k=3)
    # show first results
    result = results.documents[0]
    print(f"First score (# 1 result): {results.scores[0, 0]:.4f}")
    print(f"First result (# 1 result): {result[0]}")

if __name__ == "__main__":
    main()