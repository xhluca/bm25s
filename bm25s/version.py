from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "bm25s"
_FALLBACK_VERSION = "0.0.0"


def _discover_version() -> str:
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = _discover_version()
