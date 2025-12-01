
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="BM25S CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # MCP Subcommand
    mcp_parser = subparsers.add_parser("mcp", help="MCP Server commands")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP actions")

    # MCP Launch
    launch_parser = mcp_subparsers.add_parser("launch", help="Launch the MCP server")
    launch_parser.add_argument("-p", "--port", type=int, default=8000, help="Port to run the server on")
    launch_parser.add_argument("-d", "--index-dir", required=True, help="Path to the BM25S index directory")

    # Index Subcommand
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

    # Search Subcommand
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

    args = parser.parse_args()

    if args.command == "mcp":
        if args.mcp_command == "launch":
            try:
                from .mcp.server import main as mcp_main
                mcp_main(index_dir=args.index_dir, port=args.port)
            except ImportError:
                sys.exit(1)
            except Exception as e:
                print(f"Error launching MCP server: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            mcp_parser.print_help()
    elif args.command == "index":
        from .terminal import index_command
        index_command(args)
    elif args.command == "search":
        from .terminal import search_command
        search_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
