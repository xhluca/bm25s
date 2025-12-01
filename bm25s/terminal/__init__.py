"""
Terminal-based CLI for BM25S.

This module provides a simple command-line interface for:
- Indexing documents from CSV, TXT, JSON, or JSONL files
- Searching an index with a query

Example usage:
    bm25 index documents.csv -o my_index
    bm25 index documents.csv -u -o my_index  # saves to ~/.bm25s/indices/my_index
    bm25 search -i my_index "what is machine learning?"
    bm25 search -u "what is machine learning?"  # interactive index picker
"""

import argparse
import json
import os
import sys
from pathlib import Path


def get_user_indices_dir():
    """Get the user's BM25S indices directory (~/.bm25s/indices/)."""
    return Path.home() / ".bm25s" / "indices"


def list_user_indices():
    """List all indices in the user's indices directory."""
    indices_dir = get_user_indices_dir()
    if not indices_dir.exists():
        return []
    
    indices = []
    for item in indices_dir.iterdir():
        if item.is_dir():
            # Check if it looks like a valid index (has params.index.json)
            if (item / "params.index.json").exists():
                indices.append(item.name)
    
    return sorted(indices)


def select_index_interactive():
    """
    Show an interactive picker for selecting an index from the user directory.
    Requires 'rich' package (install with: pip install bm25s[cli])
    """
    indices = list_user_indices()
    
    if not indices:
        print("No indices found in ~/.bm25s/indices/", file=sys.stderr)
        print("Create one with: bm25 index <file> -u", file=sys.stderr)
        sys.exit(1)
    
    try:
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        # Display available indices
        table = Table(title="Available Indices", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Index Name", style="green")
        table.add_column("Path", style="dim")
        
        indices_dir = get_user_indices_dir()
        for i, name in enumerate(indices, 1):
            table.add_row(str(i), name, str(indices_dir / name))
        
        console.print()
        console.print(Panel(table, border_style="blue"))
        console.print()
        
        # Prompt for selection
        while True:
            choice = Prompt.ask(
                "[bold yellow]Select an index[/bold yellow]",
                choices=[str(i) for i in range(1, len(indices) + 1)] + indices,
                show_choices=False
            )
            
            # Check if it's a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(indices):
                    selected = indices[idx]
                    console.print(f"[green]✓[/green] Selected: [bold]{selected}[/bold]")
                    return str(indices_dir / selected)
            # Check if it's a name
            elif choice in indices:
                console.print(f"[green]✓[/green] Selected: [bold]{choice}[/bold]")
                return str(indices_dir / choice)
            
            console.print("[red]Invalid selection. Try again.[/red]")
    
    except ImportError:
        # Fallback to simple text-based selection
        print("\nAvailable indices:")
        print("-" * 40)
        indices_dir = get_user_indices_dir()
        for i, name in enumerate(indices, 1):
            print(f"  [{i}] {name}")
        print("-" * 40)
        
        while True:
            try:
                choice = input("Select an index (number or name): ").strip()
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(indices):
                        return str(indices_dir / indices[idx])
                elif choice in indices:
                    return str(indices_dir / choice)
                
                print("Invalid selection. Try again.")
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                sys.exit(1)


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
    use_user_dir = getattr(args, 'user', False)
    
    if args.output:
        if use_user_dir:
            # Save to user directory with custom name
            output_dir = get_user_indices_dir() / args.output
        else:
            output_dir = Path(args.output)
    else:
        # Default to input filename without extension + "_index"
        default_name = input_file.stem + "_index"
        if use_user_dir:
            output_dir = get_user_indices_dir() / default_name
        else:
            output_dir = Path(default_name)
    
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
    
    if use_user_dir:
        print(f"\nTo search this index, use:")
        print(f"  bm25 search -u -i {output_dir.name} \"your query\"")


def search_command(args):
    """
    Search an index with a query and print results.
    Uses the high-level API components for search.
    """
    from .. import BM25
    from ..high_level import BM25Search
    
    use_user_dir = getattr(args, 'user', False)
    index_arg = getattr(args, 'index', None)
    
    # Determine index path
    if use_user_dir:
        if index_arg:
            # Use specified index from user directory
            index_dir = get_user_indices_dir() / index_arg
        else:
            # Interactive picker
            index_path = select_index_interactive()
            index_dir = Path(index_path)
    elif index_arg:
        index_dir = Path(index_arg)
    else:
        print("Error: Must specify --index or use --user flag.", file=sys.stderr)
        sys.exit(1)
    
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
    
    # Save results to file if requested
    save_path = getattr(args, 'save', None)
    if save_path:
        output_data = {
            "query": query,
            "num_results": len(results[0]),
            "total_documents": num_docs,
            "results": results[0]
        }
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to '{save_path}'")
    
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
        help="Output directory/name for the index (default: <filename>_index)",
    )
    index_parser.add_argument(
        "-c", "--column",
        type=str,
        default=None,
        help="Column name for document text (for CSV/JSON/JSONL files)",
    )
    index_parser.add_argument(
        "-u", "--user",
        action="store_true",
        default=False,
        help="Save index to user directory (~/.bm25s/indices/)",
    )
    
    # Search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search an index with a query",
    )
    search_parser.add_argument(
        "-i", "--index",
        type=str,
        default=None,
        help="Path to the index directory (or index name if using -u)",
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
    search_parser.add_argument(
        "-s", "--save",
        type=str,
        default=None,
        help="Save results to a JSON file at the specified path",
    )
    search_parser.add_argument(
        "-u", "--user",
        action="store_true",
        default=False,
        help="Use index from user directory (~/.bm25s/indices/). Shows picker if -i not specified.",
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
