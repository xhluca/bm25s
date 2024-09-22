import json
import logging
import os
import shutil
import tempfile
from typing import Iterable, Union
from . import BM25, __version__
from .tokenization import Tokenizer

try:
    from huggingface_hub import HfApi
except ImportError:
    raise ImportError(
        "Please install the huggingface_hub package to use the HuggingFace integrations for bm25s. You can install it via `pip install huggingface_hub`."
    )

README_TEMPLATE = """---
language: en
library_name: bm25s
tags:
- bm25
- bm25s
- retrieval
- search
- lexical
---

# BM25S Index

This is a BM25S index created with the [`bm25s` library](https://github.com/xhluca/bm25s) (version `{version}`), an ultra-fast implementation of BM25. It can be used for lexical retrieval tasks.

BM25S Related Links:

* 🏠[Homepage](https://bm25s.github.io)
* 💻[GitHub Repository](https://github.com/xhluca/bm25s)
* 🤗[Blog Post](https://huggingface.co/blog/xhluca/bm25s)
* 📝[Technical Report](https://arxiv.org/abs/2407.03618)


## Installation

You can install the `bm25s` library with `pip`:

```bash
pip install "bm25s=={version}"

# Include extra dependencies like stemmer
pip install "bm25s[full]=={version}"

# For huggingface hub usage
pip install huggingface_hub
```

## Loading a `bm25s` index

You can use this index for information retrieval tasks. Here is an example:

```python
import bm25s
from bm25s.hf import BM25HF

# Load the index
retriever = BM25HF.load_from_hub("{username}/{repo_name}")

# You can retrieve now
query = "a cat is a feline"
results = retriever.retrieve(bm25s.tokenize(query), k=3)
```

## Saving a `bm25s` index

You can save a `bm25s` index to the Hugging Face Hub. Here is an example:

```python
import bm25s
from bm25s.hf import BM25HF

corpus = [
    "a cat is a feline and likes to purr",
    "a dog is the human's best friend and loves to play",
    "a bird is a beautiful animal that can fly",
    "a fish is a creature that lives in water and swims",
]

retriever = BM25HF(corpus=corpus)
retriever.index(bm25s.tokenize(corpus))

token = None  # You can get a token from the Hugging Face website
retriever.save_to_hub("{username}/{repo_name}", token=token)
```

## Advanced usage

You can leverage more advanced features of the BM25S library during `load_from_hub`:

```python
# Load corpus and index in memory-map (mmap=True) to reduce memory
retriever = BM25HF.load_from_hub("{username}/{repo_name}", load_corpus=True, mmap=True)

# Load a different branch/revision
retriever = BM25HF.load_from_hub("{username}/{repo_name}", revision="main")

# Change directory where the local files should be downloaded
retriever = BM25HF.load_from_hub("{username}/{repo_name}", local_dir="/path/to/dir")

# Load private repositories with a token:
retriever = BM25HF.load_from_hub("{username}/{repo_name}", token=token)
```

## Tokenizer

If you have saved a `Tokenizer` object with the index using the following approach:

```python
from bm25s.hf import TokenizerHF

token = "your_hugging_face_token"
tokenizer = TokenizerHF(corpus=corpus, stopwords="english")
tokenizer.save_to_hub("{username}/{repo_name}", token=token)

# and stopwords too
tokenizer.save_stopwords_to_hub("{username}/{repo_name}", token=token)
```

Then, you can load the tokenizer using the following code:

```python
from bm25s.hf import TokenizerHF

tokenizer = TokenizerHF(corpus=corpus, stopwords=[])
tokenizer.load_vocab_from_hub("{username}/{repo_name}", token=token)
tokenizer.load_stopwords_from_hub("{username}/{repo_name}", token=token)
```


## Stats

This dataset was created using the following data:

| Statistic | Value |
| --- | --- |
| Number of documents | {num_docs} |
| Number of tokens | {num_tokens} |
| Average tokens per document | {avg_tokens_per_doc} |

## Parameters

The index was created with the following parameters:

| Parameter | Value |
| --- | --- |
| k1 | `{k1}` |
| b | `{b}` |
| delta | `{delta}` |
| method | `{method}` |
| idf method | `{idf_method}` |

## Citation

To cite `bm25s`, please use the following bibtex:

```
@misc{{lu_2024_bm25s,
      title={{BM25S: Orders of magnitude faster lexical search via eager sparse scoring}}, 
      author={{Xing Han Lù}},
      year={{2024}},
      eprint={{2407.03618}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.IR}},
      url={{https://arxiv.org/abs/2407.03618}}, 
}}
```

"""


def batch_tokenize(tokenizer, texts, add_special_tokens=False):
    from tqdm.auto import tqdm

    tokenizer_kwargs = dict(
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=add_special_tokens,
        max_length=None,
    )
    tokenized = tokenizer(texts, **tokenizer_kwargs)
    output = []

    for i in tqdm(
        range(len(texts)), desc="Processing tokens (huggingface tokenizer)", leave=False
    ):
        output.append(tokenized[i].tokens)

    return output


def is_dir_empty(local_save_dir):
    """
    Check if a directory is empty or not.

    Parameters
    ----------
    local_save_dir: str
        The directory to check.

    Returns
    -------
    bool
        True if the directory is empty, False otherwise.
    """
    if not os.path.exists(local_save_dir):
        return True
    return len(os.listdir(local_save_dir)) == 0


def can_save_locally(local_save_dir, overwrite_local: bool) -> bool:
    """
    Check if it is possible to save the model to a local directory.

    Parameters
    ----------
    local_save_dir: str
        The directory to save the model to.

    overwrite_local: bool
        Whether to overwrite the existing local directory if it exists.

    Returns
    -------
    bool
        True if it is possible to save the model to the local directory, False otherwise.
    """
    # if local_save_dir is None, we cannot save locally
    if local_save_dir is None:
        return False

    # if the directory is empty, we can save locally
    if is_dir_empty(local_save_dir):
        return True

    # if we are allowed to overwrite the directory, we can save locally
    if overwrite_local:
        return True


class TokenizerHF(Tokenizer):
    def save_vocab_to_hub(
        self,
        repo_id: str,
        token: str = None,
        local_dir: str = None,
        commit_message: str = "Update tokenizer",
        overwrite_local: bool = False,
        private=True,
        **kwargs,
    ):
        """
        This function saves the tokenizer's vocab to the Hugging Face Hub.

        Parameters
        ----------
        repo_id: str
            The unique identifier of the repository to save the model to.
            The `repo_id` should be in the form of "username/repo_name".
        
        token: str
            The Hugging Face API token to use.
        
        local_dir: str
            The directory to save the model to before pushing to the Hub.
            If it is not empty and `overwrite_local` is False, it will fall
            back to saving to a temporary directory.
        
        commit_message: str
            The commit message to use when saving the model.
        
        overwrite_local: bool
            Whether to overwrite the existing local directory if it exists.
        
        kwargs: dict
            Additional keyword arguments to pass to `HfApi.upload_folder` call.
        """
        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            token=api.token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        repo_id = repo_url.repo_id

        saving_locally = can_save_locally(local_dir, overwrite_local)
        if saving_locally:
            os.makedirs(local_dir, exist_ok=True)
            save_dir = local_dir
        else:
            # save to a temporary directory otherwise
            save_dir = tempfile.mkdtemp()

        self.save_vocab(save_dir)
        # push content of the temporary directory to the repo
        api.upload_folder(
            repo_id=repo_id,
            commit_message=commit_message,
            token=api.token,
            folder_path=save_dir,
            repo_type=repo_url.repo_type,
            **kwargs,
        )
        # delete the temporary directory if it was created
        if not saving_locally:
            shutil.rmtree(save_dir)

        return repo_url
    
    def load_vocab_from_hub(
        cls,
        repo_id: str,
        revision=None,
        token=None,
        local_dir=None,
    ):
        """
        This function loads the tokenizer's vocab from the Hugging Face Hub.

        Parameters
        ----------
        repo_id: str
            The unique identifier of the repository to load the model from.
            The `repo_id` should be in the form of "username/repo_name".
        
        revision: str
            The revision of the model to load.
        
        token: str
            The Hugging Face API token to use.
        
        local_dir: str
            The local dir where the model will be stored after downloading.
        
        allow_pickle: bool
            Whether to allow pickling the model. Default is False.
        """
        api = HfApi(token=token)
        # check if the model exists
        repo_url = api.repo_info(repo_id)
        if repo_url is None:
            raise ValueError(f"Model {repo_id} not found on the Hugging Face Hub.")

        snapshot = api.snapshot_download(
            repo_id=repo_id, revision=revision, token=token, local_dir=local_dir
        )
        if snapshot is None:
            raise ValueError(f"Model {repo_id} not found on the Hugging Face Hub.")

        return cls.load_vocab(save_dir=snapshot)

    def save_stopwords_to_hub(
        self,
        repo_id: str,
        token: str = None,
        local_dir: str = None,
        commit_message: str = "Update stopwords",
        overwrite_local: bool = False,
        private=True,
        **kwargs,
    ):
        """
        This function saves the tokenizer's stopwords to the Hugging Face Hub.

        Parameters
        ----------
        repo_id: str
            The unique identifier of the repository to save the model to.
            The `repo_id` should be in the form of "username/repo_name".
        
        token: str
            The Hugging Face API token to use.
        
        local_dir: str
            The directory to save the model to before pushing to the Hub.
            If it is not empty and `overwrite_local` is False, it will fall
            back to saving to a temporary directory.
        
        commit_message: str
            The commit message to use when saving the model.
        
        overwrite_local: bool
            Whether to overwrite the existing local directory if it exists.
        
        kwargs: dict
            Additional keyword arguments to pass to `HfApi.upload_folder` call.
        """
        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            token=api.token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        repo_id = repo_url.repo_id

        saving_locally = can_save_locally(local_dir, overwrite_local)
        if saving_locally:
            os.makedirs(local_dir, exist_ok=True)
            save_dir = local_dir
        else:
            # save to a temporary directory otherwise
            save_dir = tempfile.mkdtemp()

        self.save_stopwords(save_dir)
        # push content of the temporary directory to the repo
        api.upload_folder(
            repo_id=repo_id,
            commit_message=commit_message,
            token=api.token,
            folder_path=save_dir,
            repo_type=repo_url.repo_type,
            **kwargs,
        )
        # delete the temporary directory if it was created
        if not saving_locally:
            shutil.rmtree(save_dir)

        return repo_url
    
    def load_stopwords_from_hub(
        self,
        repo_id: str,
        revision=None,
        token=None,
        local_dir=None,
    ):
        """
        This function loads the tokenizer's stopwords from the Hugging Face Hub.

        Parameters
        ----------
        repo_id: str
            The unique identifier of the repository to load the model from.
            The `repo_id` should be in the form of "username/repo_name".
        
        revision: str
            The revision of the model to load.
        
        token: str
            The Hugging Face API token to use.
        
        local_dir: str
            The local dir where the model will be stored after downloading.
        """
        api = HfApi(token=token)
        # check if the model exists
        repo_url = api.repo_info(repo_id)
        if repo_url is None:
            raise ValueError(f"Model {repo_id} not found on the Hugging Face Hub.")

        snapshot = api.snapshot_download(
            repo_id=repo_id, revision=revision, token=token, local_dir=local_dir
        )
        if snapshot is None:
            raise ValueError(f"Model {repo_id} not found on the Hugging Face Hub.")

        return self.load_stopwords(save_dir=snapshot)

class BM25HF(BM25):
    def save_to_hub(
        self,
        repo_id: str,
        token: str = None,
        local_dir: str = None,
        corpus: Iterable[Union[str, dict, list, tuple]] = None,
        private=True,
        commit_message: str = "Update BM25S model",
        overwrite_local: bool = False,
        include_readme: bool = True,
        allow_pickle: bool = False,
        **kwargs,
    ):
        """
        This function saves the BM25 model to the Hugging Face Hub.

        Parameters
        ----------

        repo_id: str
            The name of the repository to save the model to.
            the `repo_id` should be in the form of "username/repo_name".

        token: str
            The Hugging Face API token to use.

        local_dir: str
            The directory to save the model to before pushing to the Hub.
            If it is not empty and `overwrite_local` is False, it will fall
            back to saving to a temporary directory.

        corpus: Iterable[str, dict, list, tuple]
            A corpus of documents to save with the model. If it is not None,
            the corpus will be saved to the repository, as a jsonl file. If it is
            a list of string, the dictionary will have a single key "text" with the
            value being the string. If it is a list of dictionaries, apply json.dumps
            to each dictionary before saving.

        private: bool
            Whether the repository should be private or not. Default is True.

        commit_message: str
            The commit message to use when saving the model.

        overwrite_local: bool
            Whether to overwrite the existing local directory if it exists.

        include_readme: bool
            Whether to include a default README file with the model.

        allow_pickle: bool
            Whether to allow pickling the model. Default is False.

        kwargs: dict
            Additional keyword arguments to pass to `HfApi.upload_folder` call.
        """
        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            token=api.token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        repo_id = repo_url.repo_id

        username, repo_name = repo_id.split("/", 1)

        saving_locally = can_save_locally(local_dir, overwrite_local)
        if saving_locally:
            os.makedirs(local_dir, exist_ok=True)
            save_dir = local_dir
        else:
            # save to a temporary directory otherwise
            save_dir = tempfile.mkdtemp()

        self.save(save_dir, corpus=corpus, allow_pickle=allow_pickle)
        # if we include the README, write it to the directory
        if include_readme:
            num_docs = self.scores["num_docs"]
            num_tokens = self.scores["data"].shape[0]
            avg_tokens_per_doc = round(num_tokens / num_docs, 2)

            results = README_TEMPLATE.format(
                username=username,
                version=__version__,
                repo_name=repo_name,
                num_docs=num_docs,
                num_tokens=num_tokens,
                avg_tokens_per_doc=avg_tokens_per_doc,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
                method=self.method,
                idf_method=self.idf_method,
            )

            with open(os.path.join(save_dir, "README.md"), "w") as f:
                f.write(results)

        # push content of the temporary directory to the repo
        api.upload_folder(
            repo_id=repo_id,
            commit_message=commit_message,
            token=api.token,
            folder_path=save_dir,
            repo_type=repo_url.repo_type,
            **kwargs,
        )
        # delete the temporary directory if it was created
        if not saving_locally:
            shutil.rmtree(save_dir)

        return repo_url

    @classmethod
    def load_from_hub(
        cls,
        repo_name: str,
        revision=None,
        token=None,
        local_dir=None,
        load_corpus=False,
        mmap=False,
        allow_pickle=False,
    ):
        """
        This function loads the BM25 model from the Hugging Face Hub.

        Parameters
        ----------

        repo_name: str
            The name of the repository to load the model from.

        revision: str
            The revision of the model to load.

        token: str
            The Hugging Face API token to use.

        local_dir: str
            The local dir where the model will be stored after downloading.

        load_corpus: bool
            Whether to load the corpus of documents saved with the model, if present.

        mmap: bool
            Whether to memory-map the model. Default is False, which loads the index
            (and potentially corpus) into memory.

        allow_pickle: bool
            Whether to allow pickling the model. Default is False.
        """
        api = HfApi(token=token)
        # check if the model exists
        repo_url = api.repo_info(repo_name)
        if repo_url is None:
            raise ValueError(f"Model {repo_name} not found on the Hugging Face Hub.")

        snapshot = api.snapshot_download(
            repo_name, revision=revision, token=token, local_dir=local_dir
        )
        if snapshot is None:
            raise ValueError(f"Model {repo_name} not found on the Hugging Face Hub.")

        return cls.load(
            save_dir=snapshot,
            load_corpus=load_corpus,
            mmap=mmap,
            allow_pickle=allow_pickle,
        )
