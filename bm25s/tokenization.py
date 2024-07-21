import re
from typing import Any, Dict, List, Union, Callable, NamedTuple

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


from .stopwords import (
    STOPWORDS_EN,
    STOPWORDS_GERMAN,
    STOPWORDS_DUTCH,
    STOPWORDS_FRENCH,
    STOPWORDS_SPANISH,
    STOPWORDS_PORTUGUESE,
    STOPWORDS_ITALIAN,
    STOPWORDS_RUSSIAN,
    STOPWORDS_SWEDISH,
    STOPWORDS_NORWEGIAN,
    STOPWORDS_CHINESE,
)


class Tokenized(NamedTuple):
    """
    NamedTuple with two fields: ids and vocab. The ids field is a list of list of token IDs
    for each document. The vocab field is a dictionary mapping tokens to their index in the
    vocabulary.
    """

    ids: List[List[int]]
    vocab: Dict[str, int]


def convert_tokenized_to_string_list(tokenized: Tokenized) -> List[List[str]]:
    """
    Convert the token IDs back to strings using the vocab dictionary.
    """
    reverse_vocab = {v: k for k, v in tokenized.vocab.items()}

    return [
        [reverse_vocab[token_id] for token_id in doc_ids] for doc_ids in tokenized.ids
    ]


def _infer_stopwords(stopwords: Union[str, List[str]]) -> List[str]:
    if stopwords in ["english", "en", True]: # True is added to support the default
        return STOPWORDS_EN
    elif stopwords in ["german", "de"]:
        return STOPWORDS_GERMAN
    elif stopwords in ["dutch", "nl"]:
        return STOPWORDS_DUTCH
    elif stopwords in ["french", "fr"]:
        return STOPWORDS_FRENCH
    elif stopwords in ["spanish", "es"]:
        return STOPWORDS_SPANISH
    elif stopwords in ["portuguese", "pt"]:
        return STOPWORDS_PORTUGUESE
    elif stopwords in ["italian", "it"]:
        return STOPWORDS_ITALIAN
    elif stopwords in ["russian", "ru"]:
        return STOPWORDS_RUSSIAN
    elif stopwords in ["swedish", "sv"]:
        return STOPWORDS_SWEDISH
    elif stopwords in ["norwegian", "no"]:
        return STOPWORDS_NORWEGIAN
    elif stopwords in ["chinese", "zh"]:
        return STOPWORDS_CHINESE
    elif stopwords in [None, False]:
        return []
    elif isinstance(stopwords, str):
        raise ValueError(
            f"{stopwords} not recognized. Only English stopwords as default, German, Dutch, French, Spanish, Portuguese, Italian, Russian, Swedish, Norwegian, and Chinese are currently supported. "
            "Please input a list of stopwords"
        )
    else:
        return stopwords


def tokenize(
    texts,
    lower: bool = True,
    token_pattern: str = r"(?u)\b\w\w+\b",
    stopwords: Union[str, List[str]] = "english",
    stemmer: Callable = None,
    return_ids: bool = True,
    show_progress: bool = True,
    leave: bool = False,
) -> Union[List[List[str]], Tokenized]:
    """
    Tokenize a list using the same method as the scikit-learn CountVectorizer,
    and optionally apply a stemmer to the tokens or stopwords removal.

    If you provide stemmer, it must have a `stemWords` method, or be callable
    that takes a list of strings and returns a list of strings. If your stemmer
    can only be called on a single word, you can use a lambda function to wrap it,
    e.g. `lambda lst: list(map(stemmer.stem, lst))`.

    If return_ids is True, the function will return a namedtuple with: (1) the tokenized
    IDs and (2) the token_to_index dictionary. You can access the tokenized IDs using
    the `ids` attribute and the token_to_index dictionary using the `vocab` attribute,
    You can also destructure the namedtuple to get the ids and vocab_dict variables,
    e.g. `token_ids, vocab = tokenize(...)`.

    Parameters
    ----------
    texts : Union[str, List[str]]
        A list of strings to tokenize. If a single string is provided, it will be
        converted to a list with a single element.
    
    lower : bool, optional
        Whether to convert the text to lowercase before tokenization
    
    token_pattern : str, optional
        The regex pattern to use for tokenization, by default r"(?u)\b\w\w+\b"
    
    stopwords : Union[str, List[str]], optional
        The list of stopwords to remove from the text. If "english" or "en" is provided,
        the function will use the default English stopwords
    
    stemmer : Callable, optional
        The stemmer to use for stemming the tokens. It is recommended
        to use the PyStemmer library for stemming, but you can also any callable that
        takes a list of strings and returns a list of strings.
    
    return_ids : bool, optional
        Whether to return the tokenized IDs and the vocab dictionary. If False, the
        function will return the tokenized strings. If True, the function will return
        a namedtuple with the tokenized IDs and the vocab dictionary.

    show_progress : bool, optional
        Whether to show the progress bar for tokenization. If False, the function will
        not show the progress bar. If True, it will use tqdm.auto to show the progress bar.
    
    leave : bool, optional
        Whether to leave the progress bar after completion. If False, the progress bar
        will disappear after completion. If True, the progress bar will stay on the screen.

    Note
    -----
    You may pass a single string or a list of strings. If you pass a single string,
    this function will convert it to a list of strings with a single element.
    """
    if isinstance(texts, str):
        texts = [texts]

    token_pattern = re.compile(token_pattern)
    stopwords = _infer_stopwords(stopwords)

    # Step 1: Split the strings using the regex pattern
    split_fn = token_pattern.findall

    corpus_ids = []
    token_to_index = {}

    for text in tqdm(
        texts, desc="Split strings", leave=leave, disable=not show_progress
    ):
        stopwords_set = set(stopwords)
        if lower:
            text = text.lower()

        splitted = split_fn(text)
        doc_ids = []

        for token in splitted:
            if token in stopwords_set:
                continue

            if token not in token_to_index:
                token_to_index[token] = len(token_to_index)

            token_id = token_to_index[token]
            doc_ids.append(token_id)

        corpus_ids.append(doc_ids)

    # Create a list of unique tokens that we will use to create the vocabulary
    unique_tokens = list(token_to_index.keys())

    # Step 2: Stem the tokens if a stemmer is provided
    if stemmer is not None:
        if hasattr(stemmer, "stemWords"):
            stemmer_fn = stemmer.stemWords
        elif callable(stemmer):
            stemmer_fn = stemmer
        else:
            error_msg = "Stemmer must have a `stemWord` method, or be callable. For example, you can use the PyStemmer library."
            raise ValueError(error_msg)

        # Now, we use the stemmer on the token_to_index dictionary to get the stemmed tokens
        tokens_stemmed = stemmer_fn(unique_tokens)
        vocab = set(tokens_stemmed)
        vocab_dict = {token: i for i, token in enumerate(vocab)}
        stem_id_to_stem = {v: k for k, v in vocab_dict.items()}
        # We create a dictionary mapping the stemmed tokens to their index
        doc_id_to_stem_id = {
            token_to_index[token]: vocab_dict[stem]
            for token, stem in zip(unique_tokens, tokens_stemmed)
        }

        # Now, we simply need to replace the tokens in the corpus with the stemmed tokens
        for i, doc_ids in enumerate(tqdm(corpus_ids, desc="Stem Tokens", leave=leave, disable=not show_progress)):
            corpus_ids[i] = [doc_id_to_stem_id[doc_id] for doc_id in doc_ids]
    else:
        vocab_dict = token_to_index

    # Step 3: Return the tokenized IDs and the vocab dictionary or the tokenized strings
    if return_ids:
        return Tokenized(ids=corpus_ids, vocab=vocab_dict)

    else:
        # We need a reverse dictionary to convert the token IDs back to tokens
        reverse_dict = stem_id_to_stem if stemmer is not None else unique_tokens
        # We convert the token IDs back to tokens in-place
        for i, token_ids in enumerate(
            tqdm(corpus_ids, desc="Reconstructing token strings", leave=leave, disable=not show_progress)
        ):
            corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

        return corpus_ids
