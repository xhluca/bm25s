# BM25S tests

Welcome to the test suite for BM25S! This test suite is designed to test the BM25S implementation in the `bm25s` package.

## Core tests

To run the core tests (of library), simply run the following command:

```bash
python -m unittest tests/core/*.py
python -m unittest tests/stopwords/*.py
```

For numba, you have to run:

```bash
python -m unittest tests/numba/*.py
```


## Basic Comparisons

To run the basic comparison tests (with other BM25 implementations), simply run the following command:

```bash
python -m unittest tests/comparison/*.py
```

## Multiple tests

To run the core tests (of library), simply run the following command:

```bash
python -m unittest tests/core/*.py
python -m unittest tests/stopwords/*.py
python -m unittest tests/numba/*.py
python -m unittest tests/comparison/*.py
```

## Full comparison tests

To run the full comparison tests, simply run the following command:

```bash
python -m unittest tests/comparison_full/*.py
```

## Artifacts

By default, the artifacts are stored in the `./artifacts` directory. This directory is created if it does not exist. To specify the directory, you can set the `BM25_ARTIFACTS_DIR` environment variable:
 
```bash
export BM25_ARTIFACTS_DIR=/path/to/artifacts
```


## Adding new tests

First, create a new file in tests/core, tests/comparison, tests/numba, tests/stopwords, or tests/comparison_full. Then, add the following code to the file:

```python
import os
import shutil
from pathlib import Path
import unittest
import tempfile
import Stemmer  # optional: for stemming
import unittest.mock
import json

import bm25s

class TestYourName(unittest.TestCase):
    def test_your_name(self):
        # Your test code here
        pass
```

Modify the `test_your_name` function to test your code. You can use the `bm25s` package to test your code. You can also use the `unittest.mock` package to mock objects.