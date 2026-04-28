import os
import re
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

package_name = "bm25s"
base_dir = Path(__file__).resolve().parent


def _normalize_version(value):
    version = value.strip()
    if version.startswith("refs/tags/"):
        version = version.rsplit("/", 1)[-1]
    if version.startswith("v"):
        version = version[1:]
    return version


def _version_from_environment():
    for key in ("BM25S_VERSION", "RELEASE_TAG"):
        value = os.environ.get(key)
        if value:
            return _normalize_version(value)

    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        value = os.environ.get("GITHUB_REF_NAME")
        if value:
            return _normalize_version(value)

    github_ref = os.environ.get("GITHUB_REF", "")
    if github_ref.startswith("refs/tags/"):
        return _normalize_version(github_ref)

    return None


def _version_from_pkg_info(package_dir):
    pkg_info = package_dir / "PKG-INFO"
    if not pkg_info.exists():
        return None

    for line in pkg_info.read_text(encoding="utf8").splitlines():
        if line.startswith("Version:"):
            return _normalize_version(line.partition(":")[2])

    return None


def _git_describe(package_dir, *args):
    try:
        return subprocess.check_output(
            [
                "git",
                "describe",
                "--tags",
                "--match",
                "[0-9]*",
                "--match",
                "v[0-9]*",
                *args,
            ],
            cwd=package_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version_from_git(package_dir):
    exact_tag = _git_describe(package_dir, "--exact-match")
    if exact_tag:
        return _normalize_version(exact_tag)

    description = _git_describe(package_dir, "--long")
    if not description:
        return None

    match = re.match(r"(.+)-(\d+)-g([0-9a-f]+)$", description)
    if not match:
        return None

    tag, distance, sha = match.groups()
    version = _normalize_version(tag)
    if distance == "0":
        return version

    suffix = ".dev" if ".post" in version else ".post"
    return f"{version}{suffix}{distance}+g{sha}"


def _get_build_version(package_dir):
    return (
        _version_from_environment()
        or _version_from_pkg_info(package_dir)
        or _version_from_git(package_dir)
        or "0.0.0"
    )


package_version = _get_build_version(base_dir)

with open(base_dir / "README.md", encoding="utf8") as fp:
    long_description = fp.read()

extras_require = {
    "core": ["orjson", "tqdm", "PyStemmer", "numba"],
    "stem": ["PyStemmer"],
    "hf": ["huggingface_hub"],
    "dev": ["black"],
    "selection": ["jax[cpu]"],
    "indexing": ["scipy"],
    "evaluation": ["pytrec_eval"],
    "mcp": ["mcp"],
    "cli": ["rich"],
}
# Dynamically create the 'full' extra by combining all other extras
extras_require["full"] = sum(extras_require.values(), [])

setup(
    name=package_name,
    version=package_version,
    author="Xing Han Lù",
    author_email=f"{package_name}@googlegroups.com",
    url=f"https://github.com/xhluca/{package_name}",
    description=f"An ultra-fast implementation of BM25 based on sparse matrices.",
    long_description=long_description,
    packages=find_packages(include=[f"{package_name}*"]),
    package_data={},
    install_requires=['numpy'],
    entry_points={
        "console_scripts": [
            "bm25=bm25s.cli:main",
        ],
    },
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    # Cast long description to markdown
    long_description_content_type="text/markdown",
)
