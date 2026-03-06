<div align="center">

<h1>BM25</h1>

<i>A fast, simple, and high-level Python API and CLI for BM25, powered by `bm25s`.</i>

<table>
      <tr>
            <td>
                  <a href="https://github.com/xhluca/bm25s">💻 GitHub</a>
            </td>
            <td>
                  <a href="https://pypi.org/project/bm25s/">📦 bm25s</a>
            </td>
            <td>
                  <a href="https://bm25s.github.io">🏠 Homepage</a>
            </td>
      </tr>
</table>
</div>

`BM25` is a wrapper package that installs `bm25s` with its optional core dependencies, providing a simple, high-level API and a command-line interface for fast and effective text retrieval.

## Installation

Install `BM25` using pip:

```bash
pip install BM25
```

This will automatically install the highly optimized `bm25s` backend, alongside necessary dependencies for stemming (`PyStemmer`), parallelization, and CLI (`rich`).

## High Level API

If you want to quickly search on a local file, you can use the `BM25` module:

```python
import BM25

# Load a file (csv, json, jsonl, txt)
# For csv/jsonl, you can specify the column/key to use as document text
corpus = BM25.load("tests/data/dummy.csv", document_column="text")
# Index the corpus
retriever = BM25.index(corpus)

# Search
results = retriever.search(["your query here"], k=5)
for result in results[0]:
    print(result)
```

The `load` function handles file reading, while `index` handles tokenization, indexing, and provides a simple search interface.

## Command-Line Interface

The package provides a terminal-based CLI for quick indexing and searching without writing Python code.

### Indexing Documents

Create an index from a CSV, TXT, JSON, or JSONL file:

```bash
# Index a CSV file (uses first column by default)
bm25 index documents.csv -o my_index

# Index with a specific column
bm25 index documents.csv -o my_index -c text

# Index a text file (one document per line)
bm25 index documents.txt -o my_index

# Index a JSONL file
bm25 index documents.jsonl -o my_index -c content
```

If you don't specify an output directory with `-o`, the index will be saved to `<filename>_index`.

### User Directory

You can save indices to a central user directory (`~/.bm25s/indices/`) using the `-u` flag:

```bash
# Save index to ~/.bm25s/indices/my_docs
bm25 index documents.csv -u -o my_docs

# Search using the user directory
bm25 search -u -i my_docs "your query"
```

### Searching

Search an existing index with a query using `-i` (or `--index`):

```bash
# Basic search (returns top 10 results)
bm25 search -i my_index "what is machine learning?"

# Search with full path
bm25 search -i ./path/to/my_index "your query here"

# Return more results
bm25 search -i my_index "your query here" -k 20

# Save results to a JSON file
bm25 search -i my_index "your query here" -s results.json
```

### Interactive Index Picker

When using `-u` without specifying an index name, an interactive picker is displayed (requires `bm25s[cli]` which is installed by default with `BM25`):

```bash
# Interactive picker will show available indices
bm25 search -u "your query"
```

### Example Workflow

**Basic usage** (index saved to current directory):

```bash
# 1. Create a simple text file with documents
echo -e "Machine learning is a subset of AI\nDeep learning uses neural networks\nNatural language processing handles text" > docs.txt

# 2. Index the documents
bm25 index docs.txt -o my_index

# 3. Search the index
bm25 search -i my_index "what is AI?"
```

**With user directory** (indices saved to `~/.bm25s/indices/`):

```bash
# Index to user directory
bm25 index docs.txt -u -o ml_docs

# Search from user directory
bm25 search -u -i ml_docs "what is AI?"

# Or use the interactive picker
bm25 search -u "what is AI?"
```

## Flexibility

For more advanced use cases, including memory mapping, customized tokenization, hugging face integration, or using different BM25 variants, please use the underlying `bm25s` API directly. 

See the [bm25s documentation](https://github.com/xhluca/bm25s) for full details.
