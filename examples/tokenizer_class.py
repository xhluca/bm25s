"""
# Example: Retrieve from pre-built index of SciFact

This script shows how to load an index built with BM25.index and saved with BM25.save, and retrieve
the top-k results for a set of queries from the SciFact dataset, via the BEIR library.
"""
import beir.util
from beir.datasets.data_loader import GenericDataLoader
import Stemmer

import bm25s
from bm25s.utils.beir import BASE_URL
from bm25s.tokenization import Tokenizer, Tokenized


def main(data_dir="datasets", dataset="scifact"):
    # Load the queries from BEIR
    data_path = beir.util.download_and_unzip(BASE_URL.format(dataset), data_dir)
    loader = GenericDataLoader(data_folder=data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split='test')
    corpus_lst = [doc["title"] + " " + doc["text"] for doc in corpus.values()]
    queries_lst = list(queries.values())

    # Initialize the stemmer
    stemmer = Stemmer.Stemmer("english")

    # Initialize the Tokenizer with the stemmer
    tokenizer = Tokenizer(
        stemmer=stemmer,
        lower=True, # lowercase the tokens
        stopwords="english",  # or pass a list of stopwords
        splitter=r"\w+",  # by default r"(?u)\b\w\w+\b", can also be a function
    )

    # Tokenize the corpus
    corpus_tokenized = tokenizer.tokenize(
        corpus_lst, 
        update_vocab=True, # update the vocab as we tokenize
        return_as="ids"
    )

    # stream tokenizing the queries, without updating the vocabulary
    # note: this cannot return as string due to the streaming nature
    tokenizer_stream = tokenizer.streaming_tokenize(
        queries_lst, 
        update_vocab=False
    )
    query_ids = []

    for q in tokenizer_stream:
        # you can do something with the ids here, e.g. retrieve from the index
        if 1 in q:
            query_ids.append(q)

    # you can convert the ids to a Tokenized namedtuple ids and tokens
    res = tokenizer.to_tokenized_tuple(query_ids)
    assert res.ids == query_ids
    assert res.vocab == tokenizer.get_vocab_dict()
    assert isinstance(res, Tokenized)
    
    # You can also get strings
    query_strs = tokenizer.to_lists_of_strings(query_ids)
    assert isinstance(query_strs, list)
    assert isinstance(query_strs[0], list)
    assert isinstance(query_strs[0][0], str)

    # Let's see how it's all used
    retriever = bm25s.BM25()
    retriever.index(corpus_tokenized, leave_progress=False)

    # all of the above can be passed to index a bm25s model

    # e.g. using the ids directly
    results, scores = retriever.retrieve(query_ids, k=3)

    # or passing the strings
    results, scores = retriever.retrieve(query_strs, k=3)

    # or passing the Tokenized namedtuple
    results, scores = retriever.retrieve(res, k=3)
    
    # or passing a tuple of ids and vocab dict
    vocab_dict = tokenizer.get_vocab_dict()
    results, scores = retriever.retrieve((query_ids, vocab_dict), k=3)

    # Unhappy with your vocab? you can reset your tokenizer
    tokenizer.reset_vocab()


if __name__ == "__main__":
    main()

