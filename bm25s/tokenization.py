import re
from typing import Any, Dict, List, Union, Callable, NamedTuple

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable


from .stopwords import (
    STOPWORDS_EN,
    STOPWORDS_EN_PLUS,
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


class Tokenizer:
    def __init__(
        self,
        lower: bool = True,
        splitter: str = r"(?u)\b\w\w+\b",
        stopwords: Union[str, List[str]] = "english",
        stemmer: Callable = None,
    ):
        self.lower = lower
        if isinstance(splitter, str):
            splitter = re.compile(splitter).findall
        if not callable(splitter):
            raise ValueError("splitter must be a callable or a regex pattern.")

        # Exception handling for stemmer when we are using PyStemmer, which has a stemWords method
        if hasattr(stemmer, "stemWord"):
            stemmer = stemmer.stemWord
        if not callable(stemmer) and stemmer is not None:
            raise ValueError("stemmer must be callable or have a `stemWord` method.")

        self.stopwords = _infer_stopwords(stopwords)
        self.splitter = splitter
        self.stemmer = stemmer

        self.reset_vocab()

    def reset_vocab(self):
        self._word_to_wid = {}  # word -> id, e.g. "apple" -> 2
        self.word_to_stem = {}  # word -> stemmed word, e.g. "apple" -> "appl"
        self.stem_to_sid = {}  # stem -> stemmed id, e.g. "appl" -> 0
        self.word_to_id = {}  # word -> {stemmed, word} id, e.g. "apple" -> 0 (appl) or "apple" -> 2 (apple)

    def streaming_tokenize(self, texts: List[str], update_vocab: bool = True):
        """
        Tokenize a list of strings and return a generator of token IDs.

        Parameters
        ----------
        texts : List[str]
            A list of strings to tokenize.
        
        update_vocab : bool, optional
            Whether to update the vocabulary dictionary with the new tokens. If true,
            the different dictionaries making up the vocabulary will be updated with the
            new tokens. If False, the function will not update the vocabulary. Unless you have 
            a stemmer and the stemmed word is in the stem_to_sid dictionary.  If "never",
            the function will never update the vocabulary, even if the stemmed word is in
            the stem_to_sid dictionary. Note that update_vocab="if_empty" is not supported 
            in this method, only in the `tokenize` method.
        """
        stopwords_set = set(self.stopwords) if self.stopwords is not None else None
        using_stopwords = stopwords_set is not None
        using_stemmer = self.stemmer is not None

        if self.stemmer is None:
            self.word_to_id = self._word_to_wid
        
        for text in texts:
            if self.lower:
                text = text.lower()

            splitted_words = self.splitter(text)

            doc_ids = []
            for word in splitted_words:
                if word in self.word_to_id:
                    wid = self.word_to_id[word]
                    doc_ids.append(wid)
                    continue

                if using_stopwords and word in stopwords_set:
                    continue

                # We are always updating the word_to_stem mapping since even new
                # words that we have never seen before can be stemmed, with the
                # possibility that the stemmed ID is already in the stem_to_sid
                if using_stemmer:
                    if word in self.word_to_stem:
                        stem = self.word_to_stem[word]
                    else:
                        stem = self.stemmer(word)
                        self.word_to_stem[word] = stem

                    # if the stem is already in the stem_to_sid, we can just use the ID
                    # and update the word_to_id dictionary, unless update_vocab is "never"
                    # in which case we skip this word
                    if update_vocab != "never" and stem in self.stem_to_sid:
                        sid = self.stem_to_sid[stem]
                        self.word_to_id[word] = sid
                        doc_ids.append(sid)
                        
                    elif update_vocab is True:
                        sid = len(self.stem_to_sid)
                        self.stem_to_sid[stem] = sid
                        self.word_to_id[word] = sid
                        doc_ids.append(sid)
                
                if word not in self.word_to_id:
                    if update_vocab is True:
                        self._word_to_wid[word] = len(self._word_to_wid)

            yield doc_ids

    def tokenize(
        self,
        texts: List[str],
        update_vocab: Union[bool, str] = "if_empty",
        leave_progress: bool = False,
        show_progress: bool = True,
        length: int = None,
        return_as: str = "ids",
    ) -> List[List[int]]:
        """
        Tokenize a list of strings and return the token IDs.

        Parameters
        ----------
        texts : List[str]
            A list of strings to tokenize.
        
        update_vocab : bool, optional
            Whether to update the vocabulary dictionary with the new tokens. If true,
            the different dictionaries making up the vocabulary will be updated with the
            new tokens. If False, the vocabulary will not be updated unless you have a stemmer
            and the stemmed word is in the stem_to_sid dictionary. If update_vocab="if_empty", 
            the function will only update the vocabulary if it is empty, i.e. when the 
            function is called for the first time, or if the vocabulary has been reset with 
            the `reset_vocab` method. If update_vocab="never", the "word_to_id" will never 
            be updated, even if the stemmed word is in the stem_to_sid dictionary. Only use 
            this if you are sure that the stemmed words are already in the stem_to_sid dictionary.
        
        leave_progress : bool, optional
            Whether to leave the progress bar after completion. If False, the progress bar
            will disappear after completion. If True, the progress bar will stay on the screen.
        
        show_progress : bool, optional
            Whether to show the progress bar for tokenization. If False, the function will
            not show the progress bar. If True, it will use tqdm.auto to show the progress bar.
        
        length : int, optional
            The length of the texts. If None, the function will use the length of the texts.
            This is mainly used when `texts` is a generator or a stream instead of a list.
        
        return_as : str, optional
            The type of object to return by this function.
            If "tuple", this returns a Tokenized namedtuple, which contains the token IDs 
            and the vocab dictionary. 
            If "string", this return a list of lists of strings, each string being a token. 
            If "ids", this return a list of lists of integers corresponding to the token IDs,
            or stemmed IDs if a stemmer is used.

        Returns
        -------
        List[List[int]] or Generator[List[int]] or List[List[str]] or Tokenized object
            If `return_as="stream"`, a Generator[List[int]] is returned, each integer being a token ID.
            If `return_as="ids"`, a List[List[int]] is returned, each integer being a token ID.
            If `return_as="string"`, a List[List[str]] is returned, each string being a token.
            If `return_as="tuple"`, a Tokenized namedtuple is returned, with names `ids` and `vocab`.
        """
        incorrect_return_error = "return_as must be either 'tuple', 'string', 'ids', or 'stream'."
        incorrect_update_vocab_error = "update_vocab must be either True, False, 'if_empty', or 'never'."
        if return_as not in ["tuple", "string", "ids", "stream"]:
            raise ValueError(incorrect_return_error)
        
        if update_vocab not in [True, False, "if_empty", "never"]:
            raise ValueError(incorrect_update_vocab_error)

        if update_vocab == "if_empty":
            update_vocab = len(self._word_to_wid) == 0
        
        stream_fn = self.streaming_tokenize(texts=texts, update_vocab=update_vocab)

        if return_as == "stream":
            return stream_fn

        if length is None:
            length = len(texts)
        
        tqdm_kwargs = dict(
            desc="Tokenize texts",
            leave=leave_progress,
            disable=not show_progress,
            total=length,
        )

        token_ids = []
        for doc_ids in tqdm(stream_fn, **tqdm_kwargs):
            token_ids.append(doc_ids)

        if return_as == "ids":
            return token_ids
        elif return_as == "string":
            return self.to_lists_of_strings(token_ids)
        elif return_as == "tuple":
            return self.to_tokenized_tuple(token_ids)
        else:
            raise ValueError(incorrect_return_error)

    
    def get_vocab_dict(self) -> Dict[str, Any]:
        if self.stemmer is None:
            # if we are not using a stemmer, we return the word_to_id dictionary
            # which maps the words to the word IDs
            return self._word_to_wid
        else:
            # if we are using a stemmer, we return the stem_to_sid dictionary,
            # which we will use to map the stemmed words to the stemmed IDs
            return self.stem_to_sid

    def to_tokenized_tuple(self, docs: List[List[int]]) -> Tokenized:
        """
        Convert the token IDs to a Tokenized namedtuple, which contains the word IDs, or the stemmed IDs
        if a stemmer is used. The Tokenized namedtuple contains two fields: ids and vocab. The latter
        is a dictionary mapping the token IDs to the tokens, or a dictionary mapping the stemmed IDs to 
        the stemmed tokens (if a stemmer is used).
        """
        return Tokenized(ids=docs, vocab=self.get_vocab_dict())
    
    def to_lists_of_strings(self, docs: List[List[int]]) -> List[List[str]]:
        """
        Convert word IDs (or stemmed IDs if a stemmer is used) back to strings using the vocab dictionary,
        which is a dictionary mapping the word IDs to the words or a dictionary mapping the stemmed IDs 
        to the stemmed words (if a stemmer is used).
        """
        vocab = self.get_vocab_dict()
        reverse_vocab = {v: k for k, v in vocab.items()}
        return [
            [reverse_vocab[token_id] for token_id in doc] for doc in docs
        ]

def convert_tokenized_to_string_list(tokenized: Tokenized) -> List[List[str]]:
    """
    Convert the token IDs back to strings using the vocab dictionary.
    """
    reverse_vocab = {v: k for k, v in tokenized.vocab.items()}

    return [
        [reverse_vocab[token_id] for token_id in doc_ids] for doc_ids in tokenized.ids
    ]


def _infer_stopwords(stopwords: Union[str, List[str]]) -> List[str]:
    # Source of stopwords: https://github.com/nltk/nltk/blob/96ee715997e1c8d9148b6d8e1b32f412f31c7ff7/nltk/corpus/__init__.py#L315
    if stopwords in ["english", "en", True]:  # True is added to support the default
        return STOPWORDS_EN
    elif stopwords in ["english_plus", "en_plus"]:
        return STOPWORDS_EN_PLUS
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
    texts: Union[str, List[str]],
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
        for i, doc_ids in enumerate(
            tqdm(corpus_ids, desc="Stem Tokens", leave=leave, disable=not show_progress)
        ):
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
            tqdm(
                corpus_ids,
                desc="Reconstructing token strings",
                leave=leave,
                disable=not show_progress,
            )
        ):
            corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

        return corpus_ids


def _tokenize_with_vocab_exp(
    texts: Union[str, List[str]],
    lower: bool = True,
    token_pattern: str = r"(?u)\b\w\w+\b",
    stopwords: Union[str, List[str]] = "english",
    vocab_dict: dict = None,
    show_progress: bool = True,
    leave: bool = False,
) -> Tokenized:
    if isinstance(texts, str):
        texts = [texts]

    if vocab_dict is None:
        raise ValueError("vocab_dict must be provided.")

    token_pattern = re.compile(token_pattern)
    stopwords = _infer_stopwords(stopwords)

    corpus_ids = []
    stopwords_set = set(stopwords)
    for text in tqdm(
        texts, desc="Split strings", leave=leave, disable=not show_progress
    ):
        if lower:
            text = text.lower()

        splitted = token_pattern.findall(text)

        doc_ids = []

        for token in splitted:
            if token in stopwords_set:
                continue

            if token not in vocab_dict:
                continue

            doc_ids.append(vocab_dict[token])

        corpus_ids.append(doc_ids)

    # Step 3: Return the tokenized IDs and the vocab dictionary or the tokenized strings
    return corpus_ids
