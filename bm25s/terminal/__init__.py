"""
Terminal-based CLI for BM25S.

This module provides a simple command-line interface for:
- Indexing documents from CSV, TXT, JSON, or JSONL files
- Searching an index with a query

Example usage:
    bm25 index documents.csv -o my_index
    bm25 search --index=my_index "what is machine learning?"
"""

import argparse
import sys
from pathlib import Path


def index_command(args):
    """
    Index documents from a file and save the index to disk.
    Uses the high-level API for loading and indexing.
    """
    from ..high_level import load as load_documents, index as create_index
    
    input_file = Path(args.file)
    
    if not input_file.exists():
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default to input filename without extension + "_index"
        output_dir = Path(input_file.stem + "_index")
    
    print(f"Loading documents from '{input_file}'...")
    
    try:
        # Load documents using high_level.load which handles csv, json, jsonl, txt
        documents = load_documents(str(input_file), document_column=args.column)
    except Exception as e:
        print(f"Error loading documents: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(documents) == 0:
        print("Error: No documents found in the input file.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(documents)} documents.")
    print("Tokenizing and indexing...")
    
    # Use high-level index() to create the search object
    search_obj = create_index(documents)
    
    # Save the index and corpus using the underlying retriever
    print(f"Saving index to '{output_dir}'...")
    output_dir.mkdir(parents=True, exist_ok=True)
    search_obj.retriever.save(str(output_dir), corpus=documents)
    search_obj.tokenizer.save_vocab(str(output_dir))
    search_obj.tokenizer.save_stopwords(str(output_dir))
    
    print(f"Index saved successfully to '{output_dir}'")
    print(f"  - {len(documents)} documents indexed")
    print(f"  - {len(search_obj.tokenizer.get_vocab_dict())} unique tokens")


def search_command(args):
    """
    Search an index with a query and print results.
    Uses the high-level API components for search.
    """
    from .. import BM25
    from ..high_level import BM25Search
    
    index_dir = Path(args.index)
    
    if not index_dir.exists():
        print(f"Error: Index directory '{index_dir}' not found.", file=sys.stderr)
        sys.exit(1)
    
    query = args.query
    k = args.top_k
    
    print(f"Loading index from '{index_dir}'...")
    
    try:
        # Load the BM25 index with corpus
        retriever = BM25.load(str(index_dir), load_corpus=True)
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load corpus from the retriever
    corpus = retriever.corpus
    if corpus is None:
        print("Error: Corpus not found in index.", file=sys.stderr)
        sys.exit(1)
    
    # Convert corpus to list of strings if needed
    corpus_texts = []
    for doc in corpus:
        if isinstance(doc, dict):
            if 'text' in doc:
                corpus_texts.append(doc['text'])
            else:
                # Use first value
                corpus_texts.append(str(list(doc.values())[0] if doc else ""))
        else:
            corpus_texts.append(str(doc))
    
    # Create a BM25Search object with the loaded corpus
    # This will re-tokenize but ensures consistent behavior with high-level API
    search_obj = BM25Search(corpus=corpus_texts)
    
    # Perform search using high-level API
    num_docs = len(corpus_texts)
    actual_k = min(k, num_docs)
    
    results = search_obj.search([query], k=actual_k)
    
    # Print results
    print(f"\nSearch results for: \"{query}\"\n")
    print("-" * 60)
    
    for i, result in enumerate(results[0]):
        doc_text = result["document"]
        score = result["score"]
        
        # Truncate long documents for display
        max_len = 200
        if len(doc_text) > max_len:
            doc_text = doc_text[:max_len] + "..."
        
        print(f"[{i+1}] (score: {score:.4f})")
        print(f"    {doc_text}")
        print()
    
    print("-" * 60)
    print(f"Showing top {len(results[0])} of {num_docs} documents")


def create_parser():
    """Create the argument parser for the terminal CLI."""
    parser = argparse.ArgumentParser(
        prog="bm25",
        description="BM25S Terminal CLI - Index and search documents",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index subcommand
    index_parser = subparsers.add_parser(
        "index",
        help="Index documents from a file (CSV, TXT, JSON, or JSONL)",
    )
    index_parser.add_argument(
        "file",
        type=str,
        help="Path to the input file (CSV, TXT, JSON, or JSONL)",
    )
    index_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for the index (default: <filename>_index)",
    )
    index_parser.add_argument(
        "-c", "--column",
        type=str,
        default=None,
        help="Column name for document text (for CSV/JSON/JSONL files)",
    )
    
    # Search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search an index with a query",
    )
    search_parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Path to the index directory",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )
    search_parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    
    return parser


def main(args=None):
    """Main entry point for the terminal CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command == "index":
        index_command(parsed_args)
    elif parsed_args.command == "search":
        search_command(parsed_args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
