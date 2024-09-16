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

    # Tokenize the queries
    stemmer = Stemmer.Stemmer("english")
    queries_tokenized = bm25s.tokenize(queries, stemmer=stemmer)

    # Retrieve the top-k results
    retriever.activate_numba_scorer()
    results = retriever.retrieve(queries_tokenized, k=3, backend_selection="numba")
    # show first results
    result = results.documents[0]
    print(f"First score (# 1 result):{results.scores[0, 0]}")
    print(f"First result (# 1 result):\n{result[0]}")

if __name__ == "__main__":
    main()