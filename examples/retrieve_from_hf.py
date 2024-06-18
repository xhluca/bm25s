"""
# Example: Load index from Hugging Face Hub and retrieve from SciFact dataset

This shows how to load an index from the Hugging Face Hub created with BM25HF.index and 
saved with BM25HF.save_to_hub. We will retrieve the top-k results for custom queries.

To run this example, you need to install the following dependencies:

```bash
pip install bm25s[full]
```

To build an index, please refer to the `examples/index_and_upload_to_hf.py` script. You
can run this script with:

```bash
python examples/index_and_upload_to_hf.py
```

Then, run this script with:

```bash
python examples/retrieve_from_hf.py
```
"""
import os
import Stemmer

import bm25s.hf

def main(user, repo_name="bm25s-scifact-index"):
    queries = [
        "Is chemotherapy effective for treating cancer?",
        "Is Cardiac injury is common in critical cases of COVID-19?",
    ]

    # Load the BM25 index from Hugging Face Hub
    # mmap=True helps to reduce memory usage by memory-mapping the index
    # load_corpus=True loads the corpus along with the index, so you can access the documents
    retriever = bm25s.hf.BM25HF.load_from_hub(
        f"{user}/{repo_name}", load_corpus=True, mmap=True
    )

    # Tokenize the queries
    stemmer = Stemmer.Stemmer("english")
    queries_tokenized = bm25s.tokenize(queries, stemmer=stemmer)

    # Retrieve the top-k results
    results = retriever.retrieve(queries_tokenized, k=3)
    # show first results
    result = results.documents[0]
    print(f"First score (# 1 result):{results.scores[0, 0]}")
    print(f"First result (# 1 result):\n{result[0]}")

if __name__ == "__main__":
    user = os.getenv("HF_USERNAME", "write-your-username-here")
    main(user=user)