import bm25s
from bm25s.tokenization import Tokenizer

corpus = [
    "Welcome to bm25s, a library that implements BM25 in Python, allowing you to rank documents based on a query.",
    "BM25 is a widely used ranking function used for text retrieval tasks, and is a core component of search services like Elasticsearch.",
    "It is designed to be:",
    "Fast: bm25s is implemented in pure Python and leverage Scipy sparse matrices to store eagerly computed scores for all document tokens.",
    "This allows extremely fast scoring at query time, improving performance over popular libraries by orders of magnitude (see benchmarks below).",
    "Simple: bm25s is designed to be easy to use and understand.",
    "You can install it with pip and start using it in minutes.",
    "There is no dependencies on Java or Pytorch - all you need is Scipy and Numpy, and optional lightweight dependencies for stemming.",
    "Below, we compare bm25s with Elasticsearch in terms of speedup over rank-bm25, the most popular Python implementation of BM25.",
    "We measure the throughput in queries per second (QPS) on a few popular datasets from BEIR in a single-threaded setting.",
    "bm25s aims to offer a faster alternative for Python users who need efficient text retrieval.",
    "It leverages modern Python libraries and data structures for performance optimization.",
    "You can find more details in the documentation and example notebooks provided.",
    "Installation and usage guidelines are simple and accessible for developers of all skill levels.",
    "Try bm25s for a scalable and fast text ranking solution in your Python projects."
]

print(f"We have {len(corpus)} documents in the corpus.")

tokenizer = Tokenizer(splitter=lambda x: x.split())
corpus_tokens = tokenizer.tokenize(corpus, return_as="tuple")

retriever = bm25s.BM25(corpus=corpus)
retriever.index(corpus_tokens)

retriever.save("bm25s_index_readme")
tokenizer.save_vocab(save_dir="bm25s_index_readme")

# Let's reload the retriever and tokenizer and use them to retrieve documents based on a query

reloaded_retriever = bm25s.BM25.load("bm25s_index_readme", load_corpus=True)

reloaded_tokenizer = Tokenizer(splitter=lambda x: x.split())
reloaded_tokenizer.load_vocab("bm25s_index_readme")

queries = ["widely used text ranking function"]

query_tokens = reloaded_tokenizer.tokenize(queries, update_vocab=False)
results, scores = reloaded_retriever.retrieve(query_tokens, k=2)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")