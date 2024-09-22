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


def main(user, save_dir="datasets", repo_name="bm25s-scifact-testing", dataset="scifact"):
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

    # you can do the same with a tokenizer class
    tokenizer = bm25s.hf.TokenizerHF(stemmer=stemmer)
    tokenizer.tokenize(corpus_lst, update_vocab=True)
    tokenizer.save_vocab_to_hub(repo_id=f"{user}/{repo_name}", token=hf_token)

    # you can also load the retriever and tokenizer from the hub
    tokenizer_new = bm25s.hf.TokenizerHF(stemmer=stemmer, stopwords=[])
    tokenizer_new.load_vocab_from_hub(repo_id=f"{user}/{repo_name}", token=hf_token)

    # You can do the same for stopwords
    stopwords = tokenizer.stopwords
    tokenizer.save_stopwords_to_hub(repo_id=f"{user}/{repo_name}", token=hf_token)

    # you can also load the stopwords from the hub
    tokenizer_new.load_stopwords_from_hub(repo_id=f"{user}/{repo_name}", token=hf_token)
    
    print("Original stopwords:", stopwords)
    print("Reloaded stopwords:", tokenizer_new.stopwords)


if __name__ == "__main__":
    user = os.getenv("HF_USERNAME", "write-your-username-here")
    cont = input(f"Are you sure you want to upload as user '{user}'? (yes/no): ")
    if cont.lower() == "yes":
        main(user=user)