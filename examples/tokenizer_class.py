"""
# Example: Retrieve from pre-built index of SciFact

This script shows how to load an index built with BM25.index and saved with BM25.save, and retrieve
the top-k results for a set of queries from the SciFact dataset, via the BEIR library.
"""
import shutil
import tempfile
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

    # you can convert the ids to a Tokenized namedtuple ids and tokens...
    res = tokenizer.to_tokenized_tuple(query_ids)
    # ... which is equivalent to: 
    # tokenizer.tokenize(your_query_lst, return_as="tuple", update_vocab=False)

    # You can verify the results
    assert res.ids == query_ids
    assert res.vocab == tokenizer.get_vocab_dict()
    assert isinstance(res, Tokenized)

    
    # You can also get strings
    query_strs = tokenizer.decode(query_ids)
    # ... which is equivalent to: 
    # tokenizer.tokenize(your_query_lst, return_as="string", update_vocab=False)

    # let's verify the results
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

    # If you want, you can save the vocab and stopwords, it can be the same dir as your index
    your_index_dir = tempfile.mkdtemp()
    tokenizer.save_vocab(save_dir=your_index_dir)

    # Unhappy with your vocab? you can reset your tokenizer
    tokenizer.reset_vocab()


    # loading:
    new_tokenizer = Tokenizer(
        stemmer=stemmer,
        lower=True,
        stopwords=[],
        splitter=r"\w+",
    )
    print("Vocabulary size before reloading:", len(new_tokenizer.get_vocab_dict()))
    new_tokenizer.load_vocab(your_index_dir)
    print("Vocabulary size after reloading:", len(new_tokenizer.get_vocab_dict()))

    # the same can be done for stopwords
    print("stopwords before reloading:", new_tokenizer.stopwords)
    tokenizer.save_stopwords(save_dir=your_index_dir)
    new_tokenizer.load_stopwords(your_index_dir)
    print("stopwords after reloaded:", new_tokenizer.stopwords)

    # cleanup
    shutil.rmtree(your_index_dir)


if __name__ == "__main__":
    main()

