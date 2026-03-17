"""Convert bm25s/high_level/README.md to a styled single-page website."""

import os
import re
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
  <title>BM25 &ndash; Python search made simple</title>
  <meta name="description" content="The easiest way to add powerful BM25 search to your Python projects or command line." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" />
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}

    :root {{
      --bg: #0a0a0f;
      --surface: #12121a;
      --surface-2: #1a1a26;
      --border: #2a2a3a;
      --text: #e4e4ef;
      --text-muted: #8888a0;
      --accent: #6c63ff;
      --accent-soft: rgba(108, 99, 255, 0.12);
      --accent-glow: rgba(108, 99, 255, 0.25);
      --green: #22c55e;
      --radius: 12px;
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.7;
      -webkit-font-smoothing: antialiased;
    }}

    /* ---- NAV ---- */
    nav {{
      position: fixed;
      top: 0;
      width: 100%;
      z-index: 100;
      background: rgba(10, 10, 15, 0.8);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--border);
    }}
    nav .inner {{
      max-width: 960px;
      margin: 0 auto;
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    nav .logo {{
      font-weight: 700;
      font-size: 1.1rem;
      color: var(--text);
      text-decoration: none;
      letter-spacing: -0.02em;
    }}
    nav .logo span {{ color: var(--accent); }}
    nav .links {{ display: flex; gap: 1.75rem; }}
    nav .links a {{
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.875rem;
      font-weight: 500;
      transition: color 0.2s;
    }}
    nav .links a:hover {{ color: var(--text); }}

    /* ---- HERO ---- */
    .hero {{
      padding: 10rem 1.5rem 5rem;
      text-align: center;
      position: relative;
      overflow: hidden;
    }}
    .hero::before {{
      content: '';
      position: absolute;
      top: -40%;
      left: 50%;
      transform: translateX(-50%);
      width: 600px;
      height: 600px;
      background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
      pointer-events: none;
    }}
    .hero h1 {{
      font-size: 4rem;
      font-weight: 700;
      letter-spacing: -0.04em;
      margin: 0 0 1rem;
      position: relative;
    }}
    .hero .tagline {{
      font-size: 1.25rem;
      color: var(--text-muted);
      max-width: 540px;
      margin: 0 auto 2.5rem;
      position: relative;
    }}
    .hero .cta-row {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1rem;
      flex-wrap: wrap;
      position: relative;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.75rem 1.75rem;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      text-decoration: none;
      transition: all 0.2s;
    }}
    .btn-primary {{
      background: var(--accent);
      color: #fff;
    }}
    .btn-primary:hover {{ filter: brightness(1.15); transform: translateY(-1px); }}
    .btn-secondary {{
      background: var(--surface-2);
      color: var(--text);
      border: 1px solid var(--border);
    }}
    .btn-secondary:hover {{ border-color: var(--accent); }}
    .install-cmd {{
      display: inline-flex;
      align-items: center;
      gap: 0.75rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 1.25rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9rem;
      color: var(--green);
      position: relative;
      cursor: pointer;
      transition: border-color 0.2s;
    }}
    .install-cmd:hover {{ border-color: var(--accent); }}
    .install-cmd .dollar {{ color: var(--text-muted); user-select: none; }}

    /* ---- SECTIONS ---- */
    .container {{
      max-width: 960px;
      margin: 0 auto;
      padding: 0 1.5rem;
    }}

    section {{
      padding: 5rem 0;
    }}

    section h2 {{
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      margin: 0 0 0.5rem;
    }}
    section h2 .emoji {{ margin-right: 0.5rem; }}
    section .section-desc {{
      color: var(--text-muted);
      font-size: 1.05rem;
      margin: 0 0 2rem;
      max-width: 600px;
    }}

    /* ---- CARDS ---- */
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      margin-bottom: 1.5rem;
      transition: border-color 0.2s;
    }}
    .card:hover {{ border-color: var(--accent); }}
    .card h3 {{
      font-size: 1.15rem;
      font-weight: 600;
      margin: 0 0 0.75rem;
      letter-spacing: -0.01em;
    }}
    .card p {{
      color: var(--text-muted);
      margin: 0 0 1rem;
      font-size: 0.95rem;
    }}
    .card p:last-child {{ margin-bottom: 0; }}

    /* ---- CODE ---- */
    pre {{
      background: var(--surface) !important;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1.5rem !important;
      overflow-x: auto;
      margin: 1rem 0;
      position: relative;
    }}
    code {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      line-height: 1.6;
    }}
    p code {{
      background: var(--surface-2);
      border: 1px solid var(--border);
      padding: 0.15em 0.4em;
      border-radius: 4px;
      font-size: 0.85em;
    }}

    /* ---- FEATURE GRID ---- */
    .features {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1rem;
      margin-top: 2rem;
    }}
    .feature {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.5rem;
      transition: border-color 0.2s;
    }}
    .feature:hover {{ border-color: var(--accent); }}
    .feature .icon {{
      font-size: 1.5rem;
      margin-bottom: 0.75rem;
    }}
    .feature h3 {{
      font-size: 1rem;
      font-weight: 600;
      margin: 0 0 0.5rem;
    }}
    .feature p {{
      color: var(--text-muted);
      font-size: 0.875rem;
      margin: 0;
      line-height: 1.6;
    }}

    /* ---- DIVIDER ---- */
    .divider {{
      height: 1px;
      background: var(--border);
      max-width: 960px;
      margin: 0 auto;
    }}

    /* ---- FOOTER ---- */
    footer {{
      padding: 3rem 1.5rem;
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
    }}
    footer a {{ color: var(--accent); text-decoration: none; }}
    footer a:hover {{ text-decoration: underline; }}

    /* ---- RESPONSIVE ---- */
    @media (max-width: 640px) {{
      .hero h1 {{ font-size: 2.5rem; }}
      .hero .tagline {{ font-size: 1.05rem; }}
      nav .links {{ gap: 1rem; }}
      section {{ padding: 3rem 0; }}
    }}
  </style>
</head>
<body>

  <nav>
    <div class="inner">
      <a class="logo" href="#">BM<span>25</span></a>
      <div class="links">
        <a href="#install">Install</a>
        <a href="#python">Python</a>
        <a href="#cli">CLI</a>
        <a href="https://github.com/xhluca/bm25s">GitHub</a>
        <a href="https://pypi.org/project/BM25/">PyPI</a>
      </div>
    </div>
  </nav>

  <header class="hero">
    <h1>BM25</h1>
    <p class="tagline">The easiest way to add powerful search to your Python projects or command line.</p>
    <div class="cta-row">
      <div class="install-cmd" onclick="navigator.clipboard.writeText('pip install BM25')">
        <span class="dollar">$</span> pip install BM25
      </div>
      <a class="btn btn-primary" href="https://github.com/xhluca/bm25s">View on GitHub</a>
      <a class="btn btn-secondary" href="https://pypi.org/project/BM25/">PyPI</a>
    </div>
  </header>

  <div class="container">
    <div class="features">
      <div class="feature">
        <div class="icon">&#9889;</div>
        <h3>Blazing Fast</h3>
        <p>Powered by bm25s, an ultra-optimized engine. Get search-engine-grade performance in pure Python.</p>
      </div>
      <div class="feature">
        <div class="icon">&#128230;</div>
        <h3>One-Line API</h3>
        <p>Load documents, build an index, and search &mdash; each in a single line of code. No boilerplate.</p>
      </div>
      <div class="feature">
        <div class="icon">&#128187;</div>
        <h3>Built-in CLI</h3>
        <p>Index and search files directly from your terminal. Supports CSV, JSON, JSONL, and plain text.</p>
      </div>
    </div>
  </div>

  <div class="container">
    {sections}
  </div>

  <div class="divider"></div>

  <footer>
    <p>
      <strong>BM25</strong> is built on top of
      <a href="https://github.com/xhluca/bm25s">bm25s</a>.
      Released under the MIT License.
    </p>
  </footer>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/bash.min.js"></script>
  <script>hljs.highlightAll();</script>
</body>
</html>
"""

# Map README sections to IDs and clean titles
SECTION_IDS = {
    "Installation": "install",
    "Python API: 1-Line Search": "python",
    "Command-Line Interface (CLI)": "cli",
    "Going Further": "further",
}


def strip_emoji(text):
    """Remove leading emoji from section titles."""
    return re.sub(r'^[\U0001f300-\U0001f9ff\u2600-\u27bf\u2702-\u27b0\U0001fa00-\U0001faff]+\ufe0f?\s*', '', text)


def parse_sections(md_text):
    """Split README into h2 sections, skipping the header div."""
    # Remove the leading <div>...</div> hero block
    md_text = re.sub(r'^<div.*?</div>\s*', '', md_text, flags=re.DOTALL)

    # Split on ## headings
    parts = re.split(r'^## (.+)$', md_text, flags=re.MULTILINE)

    # parts[0] is the intro text before first ##
    intro = parts[0].strip()
    sections = []
    for i in range(1, len(parts), 2):
        raw_title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        clean_title = strip_emoji(raw_title)
        sec_id = SECTION_IDS.get(clean_title, clean_title.lower().replace(" ", "-"))
        sections.append((sec_id, raw_title, body))

    return intro, sections


def build_section_html(sec_id, title, body_md):
    """Render one section as a card."""
    body_html = markdown.markdown(body_md, extensions=["fenced_code", "tables"])
    return f"""
    <section id="{sec_id}">
      <h2>{title}</h2>
      {body_html}
    </section>
    <div class="divider"></div>
    """


def main():
    with open(SRC, encoding="utf-8") as f:
        md_text = f.read()

    intro, sections = parse_sections(md_text)

    intro_html = markdown.markdown(intro, extensions=["fenced_code"])
    sections_html = f'<section><p class="section-desc" style="max-width:700px;font-size:1.1rem">{intro_html}</p></section>\n<div class="divider"></div>\n'

    for sec_id, title, body in sections:
        sections_html += build_section_html(sec_id, title, body)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.format(sections=sections_html))

    print(f"Built {OUT} from {SRC}")


if __name__ == "__main__":
    main()
