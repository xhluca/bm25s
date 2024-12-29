from ast import Tuple
from pathlib import Path
import re
from typing import Any, Dict, List, Union, Callable, NamedTuple
import typing

from bm25s.utils import json_functions

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
    """
    Tokenizer class for tokenizing a list of strings and converting them to token IDs.

    Parameters
    ----------
    lower : bool, optional
        Whether to convert the text to lowercase before tokenization

    splitter : Union[str, Callable], optional
        If a string is provided, the tokenizer will interpret it as a regex pattern,
        and use the `re.compile` function to compile the pattern and use the `findall` method
        to split the text. If a callable is provided, the tokenizer will use the callable to
        split the text. The callable should take a string as input and return a list of strings.

    stopwords : Union[str, List[str]], optional
        The list of stopwords to remove from the text. If "english" or "en" is provided,
        the function will use the default English stopwords. If None or False is provided,
        no stopwords will be removed. If a list of strings is provided, the tokenizer will
        use the list of strings as stopwords.

    stemmer : Callable, optional
        The stemmer to use for stemming the tokens. It is recommended
        to use the PyStemmer library for stemming, but you can also any callable that
        takes a list of strings and returns a list of strings.
    """

    def __init__(
        self,
        lower: bool = True,
        splitter: Union[str, Callable] = r"(?u)\b\w\w+\b",
        stopwords: Union[str, List[str]] = "english",
        stemmer: Callable = None,  # type: ignore
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
        """
        Reset the vocabulary dictionaries to empty dictionaries, allowing you to
        tokenize a new set of texts without reusing the previous vocabulary.
        """
        self.word_to_stem = {}  # word -> stemmed word, e.g. "apple" -> "appl"
        self.stem_to_sid = {}  # stem -> stemmed id, e.g. "appl" -> 0
        # word -> {stemmed, unstemmed} id, e.g. "apple" -> 0 (appl) or "apple" -> 2 (apple)
        self.word_to_id = {}

    def save_vocab(self, save_dir: str, vocab_name: str = "vocab.tokenizer.json"):
        """
        Save the vocabulary dictionaries to a file. The file is saved in JSON format.

        Parameters
        ----------
        save_dir : str
            The directory where the vocabulary file is saved.
        
        vocab_name : str, optional
            The name of the vocabulary file. Default is "vocab.tokenizer.json". Make
            sure to not use the same name as the vocab.index.json file saved by the BM25
            model, as it will overwrite the vocab.index.json file and cause errors.
        """
        save_dir: Path = Path(save_dir)
        path = save_dir / vocab_name

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding='utf-8') as f:
            d = {
                "word_to_stem": self.word_to_stem,
                "stem_to_sid": self.stem_to_sid,
                "word_to_id": self.word_to_id,
            }
            f.write(json_functions.dumps(d, ensure_ascii=False))
        
    def load_vocab(self, save_dir: str, vocab_name: str = "vocab.tokenizer.json"):
        """
        Load the vocabulary dictionaries from a file. The file should be saved in JSON format.

        Parameters
        ----------
        save_dir : str
            The directory where the vocabulary file is saved.
        
        vocab_name : str, optional
            The name of the vocabulary file.
        
        Note
        ----
        The vocabulary file should be saved in JSON format, with the following keys:
        - word_to_stem: a dictionary mapping words to their stemmed words
        - stem_to_sid: a dictionary mapping stemmed words to their stemmed IDs
        - word_to_id: a dictionary mapping words to their word
        """
        path = Path(save_dir) / vocab_name

        with open(path, "r", encoding='utf-8') as f:
            d = json_functions.loads(f.read())
            self.word_to_stem = d["word_to_stem"]
            self.stem_to_sid = d["stem_to_sid"]
            self.word_to_id = d["word_to_id"]
    
    def save_stopwords(self, save_dir: str, stopwords_name: str = "stopwords.tokenizer.json"):
        """
        Save the stopwords to a file. The file is saved in JSON format.

        Parameters
        ----------
        save_dir : str
            The directory where the stopwords file is saved.
        
        stopwords_name : str, optional
            The name of the stopwords file. Default is "stopwords.tokenizer.json".
        """
        save_dir: Path = Path(save_dir)
        path = save_dir / stopwords_name

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json_functions.dumps(self.stopwords))
    
    def load_stopwords(self, save_dir: str, stopwords_name: str = "stopwords.tokenizer.json"):
        """
        Load the stopwords from a file. The file should be saved in JSON format.

        Parameters
        ----------
        save_dir : str
            The directory where the stopwords file is saved.
        
        stopwords_name : str, optional
            The name of the stopwords file.
        """
        path = Path(save_dir) / stopwords_name

        with open(path, "r") as f:
            self.stopwords = json_functions.loads(f.read())

    def streaming_tokenize(
        self, texts: List[str], update_vocab: Union[bool, str] = True, allow_empty: bool = True
    ):
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
        
        allow_empty : bool, optional
            Whether to allow the splitter to return an empty string. If False, the splitter 
            will return an empty list, which may cause issues if the tokenizer is not expecting
            an empty list. If True, the splitter will return a list with a single empty string.
        """
        stopwords_set = set(self.stopwords) if self.stopwords is not None else None
        using_stopwords = stopwords_set is not None
        using_stemmer = self.stemmer is not None
            
        if allow_empty is True and update_vocab is True and "" not in self.word_to_id:
            idx = max(self.word_to_id.values(), default=-1) + 1
            self.word_to_id[""] = idx
            
            if using_stemmer:
                if "" not in self.word_to_stem:
                    self.word_to_stem[""] = ""
                if "" not in self.stem_to_sid:
                    self.stem_to_sid[""] = idx
        
        for text in texts:
            if self.lower:
                text = text.lower()

            splitted_words = list(self.splitter(text))

            if allow_empty is True and len(splitted_words) == 0:
                splitted_words = [""]
            
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
                else:
                    # if we are not using a stemmer, we can just update the word_to_id
                    # directly rather than going through the stem_to_sid dictionary
                    if update_vocab is True and word not in self.word_to_id:
                        wid = len(self.word_to_id)
                        self.word_to_id[word] = wid
                        doc_ids.append(wid)

            if len(doc_ids) == 0 and allow_empty is True and "" in self.word_to_id:
                doc_ids = [self.word_to_id[""]]
            
            yield doc_ids

    def tokenize(
        self,
        texts: List[str],
        update_vocab: Union[bool, str] = "if_empty",
        leave_progress: bool = False,
        show_progress: bool = True,
        length: Union[int, None] = None,
        return_as: str = "ids",
        allow_empty: bool = True,
    ) -> Union[List[List[int]], List[List[str]], typing.Generator, Tokenized]:
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
            The length of the texts. If None, the function will call `len(texts)` to get the length.
            This is mainly used when `texts` is a generator or a stream instead of a list, in which case
            `len(texts)` will raise a TypeError, and you need to provide the length manually.

        return_as : str, optional
            The type of object to return by this function.
            If "tuple", this returns a Tokenized namedtuple, which contains the token IDs
            and the vocab dictionary.
            If "string", this return a list of lists of strings, each string being a token.
            If "ids", this return a list of lists of integers corresponding to the token IDs,
            or stemmed IDs if a stemmer is used.
        
        allow_empty : bool, optional
            Whether to allow the splitter to return an empty string. If False, the splitter 
            will return an empty list, which may cause issues if the tokenizer is not expecting
            an empty list. If True, the splitter will return a list with a single empty string.

        Returns
        -------
        List[List[int]] or Generator[List[int]] or List[List[str]] or Tokenized object
            If `return_as="stream"`, a Generator[List[int]] is returned, each integer being a token ID.
            If `return_as="ids"`, a List[List[int]] is returned, each integer being a token ID.
            If `return_as="string"`, a List[List[str]] is returned, each string being a token.
            If `return_as="tuple"`, a Tokenized namedtuple is returned, with names `ids` and `vocab`.
        """
        incorrect_return_error = (
            "return_as must be either 'tuple', 'string', 'ids', or 'stream'."
        )
        incorrect_update_vocab_error = (
            "update_vocab must be either True, False, 'if_empty', or 'never'."
        )
        if return_as not in ["tuple", "string", "ids", "stream"]:
            raise ValueError(incorrect_return_error)

        if update_vocab not in [True, False, "if_empty", "never"]:
            raise ValueError(incorrect_update_vocab_error)

        if update_vocab == "if_empty":
            update_vocab = len(self.word_to_id) == 0

        stream_fn = self.streaming_tokenize(texts=texts, update_vocab=update_vocab, allow_empty=allow_empty)

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
            return self.decode(token_ids)
        elif return_as == "tuple":
            return self.to_tokenized_tuple(token_ids)
        else:
            raise ValueError(incorrect_return_error)

    def get_vocab_dict(self) -> Dict[str, Any]:
        if self.stemmer is None:
            # if we are not using a stemmer, we return the word_to_id dictionary
            # which maps the words to the word IDs
            return self.word_to_id
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

    def decode(self, docs: List[List[int]]) -> List[List[str]]:
        """
        Convert word IDs (or stemmed IDs if a stemmer is used) back to strings using the vocab dictionary,
        which is a dictionary mapping the word IDs to the words or a dictionary mapping the stemmed IDs
        to the stemmed words (if a stemmer is used).

        Parameters
        ----------
        docs : List[List[int]]
            A list of lists of word IDs or stemmed IDs.

        Returns
        -------
        List[List[str]]
            A list of lists of strings, each string being a word or a stemmed word if a stemmer is used.
        """
        vocab = self.get_vocab_dict()
        reverse_vocab = {v: k for k, v in vocab.items()}
        return [[reverse_vocab[token_id] for token_id in doc] for doc in docs]


def convert_tokenized_to_string_list(tokenized: Tokenized) -> List[List[str]]:
    """
    Convert the token IDs back to strings using the vocab dictionary.
    """
    reverse_vocab = {v: k for k, v in tokenized.vocab.items()}

    return [
        [reverse_vocab[token_id] for token_id in doc_ids] for doc_ids in tokenized.ids
    ]


def _infer_stopwords(stopwords: Union[str, List[str]]) -> Union[List[str], tuple]:
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
    stemmer: Callable = None,  # type: ignore
    return_ids: bool = True,
    show_progress: bool = True,
    leave: bool = False,
    allow_empty: bool = True,
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

    allow_empty : bool, optional
        Whether to allow the splitter to return an empty string. If False, the splitter 
        will return an empty list, which may cause issues if the tokenizer is not expecting
        an empty list. If True, the splitter will return a list with a single empty string.
    Note
    -----
    You may pass a single string or a list of strings. If you pass a single string,
    this function will convert it to a list of strings with a single element.
    """
    if isinstance(texts, str):
        texts = [texts]

    split_fn = re.compile(token_pattern).findall
    stopwords = _infer_stopwords(stopwords)

    # Step 1: Split the strings using the regex pattern
    corpus_ids = []
    token_to_index = {}

    for text in tqdm(
        texts, desc="Split strings", leave=leave, disable=not show_progress
    ):  
        stopwords_set = set(stopwords)
        if lower:
            text = text.lower()

        splitted = split_fn(text)

        if allow_empty is False and len(splitted) == 0:
            splitted = [""]
        
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
