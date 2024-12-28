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

## Full tests

To run the full tests, simply run the following command:

```bash
python -m unittest tests/full/*.py
```

## Artifacts

By default, the artifacts are stored in the `./artifacts` directory. This directory is created if it does not exist. To specify the directory, you can set the `BM25_ARTIFACTS_DIR` environment variable:
 
```bash
export BM25_ARTIFACTS_DIR=/path/to/artifacts
```
