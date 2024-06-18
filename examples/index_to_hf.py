"""
# Example: Indexing BEIR dataset and upload to Hugging Face Hub

This will show how to index a dataset from BEIR and upload it to the Hugging Face Hub.

To run this example, you need to install the following dependencies:

```bash
pip install beir bm25s[full]
```

Make sure to replace `write-your-username-here` with your Hugging Face username,
or set the `HF_USERNAME` environment variable.

Then, run with:

```
export HF_USERNAME="write-your-username-here"
export HF_TOKEN="your-hf-token"
python examples/index_and_upload_to_hf.py
```
"""
import os
import beir.util
from beir.datasets.data_loader import GenericDataLoader
import Stemmer

import bm25s.hf
from bm25s.utils.beir import BASE_URL


def main(user, save_dir="datasets", repo_name="bm25s-scifact-index", dataset="scifact"):
    # First, use the beir library to download the dataset, and process it
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), save_dir)
    corpus, _, __ = GenericDataLoader(data_folder=data_path).load(split="test")
    corpus_records = [
        {'id': k, 'title': v["title"], 'text': v["text"]} for k, v in corpus.items()
    ]
    corpus_lst = [r["title"] + " " + r["text"] for r in corpus_records]

    # We will use the snowball stemmer from the PyStemmer library and tokenize the corpus
    stemmer = Stemmer.Stemmer("english")
    corpus_tokenized = bm25s.tokenize(corpus_lst, stemmer=stemmer)

    # We create a BM25 retriever, index the corpus, and save to Hugging Face Hub
    retriever = bm25s.hf.BM25HF()
    retriever.index(corpus_tokenized)

    hf_token = os.getenv("HF_TOKEN")
    retriever.save_to_hub(repo_id=f"{user}/{repo_name}", token=hf_token, corpus=corpus_records)

if __name__ == "__main__":
    user = os.getenv("HF_USERNAME", "write-your-username-here")
    cont = input(f"Are you sure you want to upload as user '{user}'? (yes/no): ")
    if cont.lower() == "yes":
        main(user=user)