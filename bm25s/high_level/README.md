<div align="center">

<h1>BM25</h1>

<i>The easiest way to add powerful search to your Python projects or command line.</i>

<table>
      <tr>
            <td>
                  <a href="https://github.com/xhluca/bm25s">💻 GitHub</a>
            </td>
            <td>
                  <a href="https://pypi.org/project/BM25/">📦 PyPI</a>
            </td>
            <td>
                  <a href="https://bm25-python.github.io">🏠 Homepage</a>
            </td>
      </tr>
</table>
</div>

**BM25** is a famous algorithm used by search engines (like Elasticsearch) to find the most relevant documents for a given search query. It works by matching keywords and scoring documents based on how often those words appear.

This package provides a dead-simple, beginner-friendly way to use BM25 in Python. Under the hood, it is powered by [`bm25s`](https://github.com/xhluca/bm25s), an ultra-fast, highly optimized library. By installing `BM25`, you get all the performance benefits of `bm25s` (including speedups and stemming) with a streamlined, 1-line API and a beautiful command-line interface.

## 🛠️ Installation

Get started in seconds with pip:

```bash
pip install BM25
```

*This automatically installs the optimized `bm25s` backend, along with necessary dependencies for better search quality (`PyStemmer`) and a colorful terminal experience (`rich`).*

## 🐍 Python API: 1-Line Search

If you want to quickly build a search engine over a local file or a list of texts, the `BM25` module makes it incredibly easy.

```python
import BM25

# 1. Load your documents (supports .csv, .json, .jsonl, .txt)
# For csv/jsonl, you can specify which column/key holds the text
corpus = BM25.load("documents.csv", document_column="text")

# 2. Build the search index
retriever = BM25.index(corpus)

# 3. Search!
queries = ["how to learn python", "best search algorithms"]
results = retriever.search(queries, k=5) # Get top 5 results

# Print the top results for the first query
for result in results[0]:
    print(f"Score: {result['score']:.2f} | Document: {result['document']}")
```

The `load` function handles reading your files, while `index` automatically takes care of text processing (tokenization, stemming) and creating the searchable index.

## 💻 Command-Line Interface (CLI)

Don't want to write code? The `BM25` package comes with a built-in terminal app for instant indexing and searching.

### Step 1: Index your documents
Turn any text, CSV, or JSON file into a search index.

```bash
# Index a simple text file (one document per line)
bm25 index documents.txt -o my_index

# Index a CSV file using a specific column for the text
bm25 index documents.csv -o my_index -c text
```

### Step 2: Search
Query your newly created index directly from the terminal.

```bash
# Basic search (returns top 10 results)
bm25 search -i my_index "what is machine learning?"

# Return more results and save them to a file
bm25 search -i my_index "your query here" -k 20 -s results.json
```

### 🌟 Pro-tip: The User Directory
You can save indices to a central user directory (`~/.bm25s/indices/`) so you can search them from anywhere on your computer without remembering file paths.

```bash
# Save to the central directory using the -u flag
bm25 index documents.csv -u -o my_docs

# Search interactively! Just type this, and a menu will let you pick your index:
bm25 search -u "what is AI?"
```

## 🚀 Going Further

The `BM25` package is designed to be simple and get out of your way. But if you find yourself needing more advanced features—like saving/loading models, integrating with Hugging Face, tweaking the math behind the algorithm, or handling massive millions-of-documents datasets—you already have the tools!

You can drop down to the underlying `bm25s` library anytime. Check out the [bm25s documentation](https://github.com/xhluca/bm25s) for full details on advanced usage.
