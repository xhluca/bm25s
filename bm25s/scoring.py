from collections import Counter
import math

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


def _calculate_doc_freqs(
    corpus_tokens, unique_tokens, show_progress=True, leave_progress=False
) -> dict:
    """
    Document Frequency, aka DF, is the number of documents that contain a specific token.
    This function return a dictionary with the document frequency of each token, which is
    why it is called `doc_frequencies`.
    """
    unique_tokens = set(unique_tokens)

    # Now that we have all the unique tokens, we can count the number of
    # documents that contain each token
    doc_frequencies = {token: 0 for token in unique_tokens}

    for doc_tokens in tqdm(
        corpus_tokens,
        leave=leave_progress,
        disable=not show_progress,
        desc="BM25S Count Tokens",
    ):

        # get intersection of unique tokens and the tokens in the document
        shared_tokens = unique_tokens.intersection(doc_tokens)

        # for each token in the document, we increment the count of documents
        # This is a simple way to count the number of documents that contain each token
        for token in shared_tokens:
            doc_frequencies[token] += 1

    return doc_frequencies


def _build_idf_array(
    doc_frequencies: dict,
    n_docs: int,
    compute_idf_fn: callable = None,
    dtype="float32",
) -> np.ndarray:
    n_vocab = len(doc_frequencies)
    idf_array = np.zeros(n_vocab, dtype=dtype)

    for token_id, df in doc_frequencies.items():
        idf_array[token_id] = compute_idf_fn(df, N=n_docs)

    return idf_array


def _build_nonoccurrence_array(
    doc_frequencies: dict,
    n_docs: int,
    compute_idf_fn: callable,
    calculate_tfc_fn: callable,
    l_d,
    l_avg,
    k1,
    b,
    delta,
    dtype="float32",
) -> np.ndarray:
    """
    The non-occurrence array is used to store the idf score for tokens that do not occur in the
    document. This is useful for BM25L and BM25+ variants, where we need to calculate the idf
    score for tokens that do not occur in the document, which will be used to calculate the
    final score.

    The nonoccurence array has length |V|, where V is the set of unique tokens in the corpus.

    The `compute_idf_fn` is the function to calculate the idf score for a token that does not occur
    in the document. The `calculate_tfc_fn` is the function to calculate the term frequency component
    of the BM25 score, which is used to calculate the final score for tokens that do not occur in the
    document.
    """
    n_vocab = len(doc_frequencies)
    nonoccurrence_array = np.zeros(n_vocab, dtype=dtype)

    for token_id, df in doc_frequencies.items():
        idf = compute_idf_fn(df, N=n_docs)
        tfc = calculate_tfc_fn(
            tf_array=0, l_d=l_d, l_avg=l_avg, k1=k1, b=b, delta=delta
        )
        nonoccurrence_array[token_id] = idf * tfc

    return nonoccurrence_array


def _score_tfc_robertson(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    # idf component is given by the idf_array
    # we calculate the term-frequency component (tfc)
    return tf_array / (k1 * ((1 - b) + b * l_d / l_avg) + tf_array)


def _score_tfc_lucene(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return _score_tfc_robertson(tf_array, l_d, l_avg, k1, b)


def _score_tfc_atire(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    # idf component is given by the idf_array
    # we calculate the term-frequency component (tfc)
    return (tf_array * (k1 + 1)) / (tf_array + k1 * (1 - b + b * l_d / l_avg))


def _score_tfc_bm25l(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    c_array = tf_array / (1 - b + b * l_d / l_avg)
    return ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)


def _score_tfc_bm25plus(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    num = (k1 + 1) * tf_array
    den = k1 * (1 - b + b * l_d / l_avg) + tf_array
    return (num / den) + delta


def _select_tfc_scorer(method) -> callable:
    if method == "robertson":
        return _score_tfc_robertson
    elif method == "lucene":
        return _score_tfc_lucene
    elif method == "atire":
        return _score_tfc_atire
    elif method == "bm25l":
        return _score_tfc_bm25l
    elif method == "bm25+":
        return _score_tfc_bm25plus
    else:
        error_msg = f"Invalid score_tfc value: {method}. Choose from 'robertson', 'lucene', 'atire'."
        raise ValueError(error_msg)


def _score_idf_robertson(df, N, allow_negative=False):
    """
    Computes the inverse document frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    inner = (N - df + 0.5) / (df + 0.5)
    if not allow_negative and inner < 1:
        inner = 1

    return math.log(inner)


def _score_idf_lucene(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


def _score_idf_atire(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log(N / df)


def _score_idf_bm25l(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log((N + 1) / (df + 0.5))


def _score_idf_bm25plus(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log((N + 1) / df)


def _select_idf_scorer(method) -> callable:
    if method == "robertson":
        return _score_idf_robertson
    elif method == "lucene":
        return _score_idf_lucene
    elif method == "atire":
        return _score_idf_atire
    elif method == "bm25l":
        return _score_idf_bm25l
    elif method == "bm25+":
        return _score_idf_bm25plus
    else:
        error_msg = f"Invalid score_idf_inner value: {method}. Choose from 'robertson', 'lucene', 'atire', 'bm25l', 'bm25+'."
        raise ValueError(error_msg)


def _get_counts_from_token_ids(token_ids, dtype, int_dtype):
    token_counter = Counter(token_ids)
    voc_ind = np.array(list(token_counter.keys()), dtype=int_dtype)
    tf_array = np.array(list(token_counter.values()), dtype=dtype)

    return voc_ind, tf_array


def _build_scores_and_indices_for_matrix(
    corpus_token_ids,
    idf_array,
    avg_doc_len,
    doc_frequencies,
    k1,
    b,
    delta,
    nonoccurrence_array,
    method="robertson",
    dtype="float32",
    int_dtype="int32",
    show_progress=True,
    leave_progress=False,
):
    array_size = sum(doc_frequencies.values())

    # We create 3 arrays to store the scores, document indices, and vocabulary indices
    # The length is at most n_tokens, remaining elements will be truncated at the end
    scores = np.empty(array_size, dtype=dtype)
    doc_indices = np.empty(array_size, dtype=int_dtype)
    voc_indices = np.empty(array_size, dtype=int_dtype)

    calculate_tfc = _select_tfc_scorer(method)

    i = 0
    for doc_idx, token_ids in enumerate(
        tqdm(
            corpus_token_ids,
            desc="BM25S Compute Scores",
            disable=not show_progress,
            leave=leave_progress,
        )
    ):
        doc_len = len(token_ids)

        # Get the term frequency array for the document
        # Note: tokens might contain duplicates, we use Counter to get the term freq
        voc_ind_doc, tf_array = _get_counts_from_token_ids(
            token_ids, dtype=dtype, int_dtype=int_dtype
        )

        # Calculate the BM25 score for each token in the document
        tfc = calculate_tfc(
            tf_array=tf_array, l_d=doc_len, l_avg=avg_doc_len, k1=k1, b=b, delta=delta
        )
        idf = idf_array[voc_ind_doc]
        scores_doc = idf * tfc

        # If the method is uses a non-occurrence score array, then we need to subtract
        # the non-occurrence score from the scores
        if method in ("bm25l", "bm25+"):
            scores_doc -= nonoccurrence_array[voc_ind_doc]

        # Update the arrays with the new scores, document indices, and vocabulary indices
        doc_len = len(scores_doc)
        start, end = i, i + doc_len
        i = end

        doc_indices[start:end] = doc_idx
        voc_indices[start:end] = voc_ind_doc
        scores[start:end] = scores_doc

    return scores, doc_indices, voc_indices


def _compute_relevance_from_scores_legacy(
    data, indptr, indices, num_docs, query_tokens_ids, dtype
):
    """
    The legacy implementation of the `_compute_relevance_from_scores` function. This may
    be faster than the new implementation for some cases, but it cannot benefit from
    numba acceleration, as it uses python lists. This function is kept for reference
    and comparison purposes.
    """
    # First, we use the query_token_ids to select the relevant columns from the score_matrix
    query_tokens_ids = np.array(query_tokens_ids, dtype=int)
    indptr_starts = indptr[query_tokens_ids]
    indptr_ends = indptr[query_tokens_ids + 1]

    scores_lists = []
    indices_lists = []

    for i, (start, end) in enumerate(zip(indptr_starts, indptr_ends)):
        scores_lists.append(data[start:end])
        indices_lists.append(indices[start:end])

    # combine the lists into a single array

    scores = np.zeros(num_docs, dtype=dtype)
    if len(scores_lists) == 0:
        return scores

    scores_flat = np.concatenate(scores_lists)
    indices_flat = np.concatenate(indices_lists)
    np.add.at(scores, indices_flat, scores_flat)

    return scores

def _compute_relevance_from_scores_jit_ready(
    data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    num_docs: int,
    query_tokens_ids: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    This internal static function calculates the relevance scores for a given query,
    by using the BM25 scores that have been precomputed in the BM25 eager index.
    This version is ready for JIT compilation with numba, but is slow if not compiled.
    """
    indptr_starts = indptr[query_tokens_ids]
    indptr_ends = indptr[query_tokens_ids + 1]

    scores = np.zeros(num_docs, dtype=dtype)
    for i in range(len(query_tokens_ids)):
        start, end = indptr_starts[i], indptr_ends[i]
        # The following code is slower with numpy, but faster after JIT compilation
        for j in range(start, end):
            scores[indices[j]] += data[j]

    return scores