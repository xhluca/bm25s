"""Convert bm25s/high_level/README.md to a styled index.html for GitHub Pages."""

import os
import markdown

SRC = os.path.join("bm25s", "high_level", "README.md")
OUT_DIR = "_site"
OUT = os.path.join(OUT_DIR, "index.html")

TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BM25 – Python search made simple</title>
  <meta name="description" content="The easiest way to add powerful BM25 search to your Python projects or command line." />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.6.1/github-markdown.min.css" />
  <style>
    body {{
      background: #0d1117;
      color: #e6edf3;
      margin: 0;
      padding: 2rem 1rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
    }}
    .markdown-body {{
      max-width: 860px;
      margin: 0 auto;
      background: #161b22;
      padding: 2.5rem;
      border-radius: 12px;
      border: 1px solid #30363d;
    }}
    /* dark theme overrides */
    .markdown-body {{
      color: #e6edf3;
    }}
    .markdown-body a {{
      color: #58a6ff;
    }}
    .markdown-body code {{
      background: #1c2128;
      border: 1px solid #30363d;
    }}
    .markdown-body pre {{
      background: #1c2128 !important;
      border: 1px solid #30363d;
    }}
    .markdown-body pre code {{
      border: none;
    }}
    .markdown-body h1, .markdown-body h2, .markdown-body h3 {{
      border-bottom-color: #30363d;
    }}
    .markdown-body table th, .markdown-body table td {{
      border-color: #30363d;
    }}
    .markdown-body hr {{
      background-color: #30363d;
    }}
    .markdown-body blockquote {{
      border-left-color: #30363d;
      color: #8b949e;
    }}
  </style>
</head>
<body>
  <article class="markdown-body">
    {content}
  </article>
</body>
</html>
"""


def main():
    with open(SRC, encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "toc"],
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.format(content=html_body))

    print(f"Built {OUT} from {SRC}")


if __name__ == "__main__":
    main()
