from setuptools import setup
import os
import sys

# Change to the current directory to avoid issues if run from elsewhere
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

package_name = "BM25"
version = {}
with open(os.path.join("..", "version.py"), encoding="utf8") as fp:
    exec(fp.read(), version)

with open("README.md", encoding="utf8") as fp:
    long_description = fp.read()

setup(
    name=package_name,
    version=version["__version__"],
    author="Xing Han Lù",
    author_email="bm25s@googlegroups.com",
    url="https://github.com/xhluca/bm25s/tree/main/bm25s/high_level",
    description="A simple high-level API and CLI for BM25.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["BM25"],
    package_dir={"BM25": "."},
    install_requires=[
        f"bm25s[core,cli]=={version['__version__']}",
    ],
    entry_points={
        "console_scripts": [
            "bm25=bm25s.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)