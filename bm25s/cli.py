
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
    launch_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    launch_parser.add_argument("--index-dir", required=True, help="Path to the BM25S index directory")

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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
