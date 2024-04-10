from setuptools import setup, find_packages

package_name = "bm25s"
version = {}
with open(f"{package_name}/version.py") as fp:
    exec(fp.read(), version)

with open("README.md") as fp:
    long_description = fp.read()

extras_require = {
    "extra": ["PyStemmer", "tqdm"],
    "dev": ["black"],
}
# Dynamically create the 'all' extra by combining all other extras
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name=package_name,
    version=version["__version__"],
    author="Xing Han Lù",
    author_email=f"{package_name}@googlegroups.com",
    url=f"https://github.com/xhluca/{package_name}",
    description="An ultra-fast implementation of BM25 based on sparse matrices",
    long_description=long_description,
    packages=find_packages(include=[f"{package_name}*"]),
    package_data={},
    install_requires=['scipy', 'numpy'],
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