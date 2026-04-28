from setuptools import setup
import os
import re
import subprocess
from pathlib import Path

# Change to the current directory to avoid issues if run from elsewhere
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

package_name = "BM25"
base_dir = Path(current_dir)


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

with open("README.md", encoding="utf8") as fp:
    long_description = fp.read()

setup(
    name=package_name,
    version=package_version,
    author="Xing Han Lù",
    author_email="bm25s@googlegroups.com",
    url="https://github.com/xhluca/bm25s/tree/main/bm25s/high_level",
    description="A simple high-level API and CLI for BM25.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["BM25"],
    package_dir={"BM25": "."},
    install_requires=[
        f"bm25s[core,cli]=={package_version}",
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
