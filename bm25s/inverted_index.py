import numpy as np

from .tokenization import Tokenized, convert_tokenized_to_string_list
from tqdm.auto import tqdm

def build_inverted_index(corpus_tokens: Tokenized) -> dict:
    ids = corpus_tokens.ids
    vocab = corpus_tokens.vocab

    inverted_index = {i: [] for i in range(len(vocab))}
    for doc_id, doc in tqdm(enumerate(ids)):
        for token_id in doc:
            inverted_index[token_id].append(doc_id)
    
    return inverted_index

def select_relevant_indices(queries: list, corpus_vocab, inverted_index: dict, k=None) -> list:
    if isinstance(queries, Tokenized):
        queries = convert_tokenized_to_string_list(queries)
    
    query_ids = [
        [corpus_vocab[token] for token in query if token in corpus_vocab]
        for query in queries
    ]
    
    relevant_indices = []

    for query in tqdm(query_ids, leave=False, desc="Selecting Relevant Indices"):
        relevant_docs = []
        for token_id in query:
            relevant_docs.extend(inverted_index.get(token_id, []))
        rel = np.unique(relevant_docs).astype(np.int64)
        relevant_indices.append(rel)
    
    return relevant_indices