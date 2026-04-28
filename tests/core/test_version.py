import bm25s.version as version_module


def test_discover_version_uses_package_metadata(monkeypatch):
    monkeypatch.setattr(version_module, "version", lambda name: "1.2.3")

    assert version_module._discover_version() == "1.2.3"


def test_discover_version_falls_back_when_distribution_is_missing(monkeypatch):
    def missing_version(name):
        raise version_module.PackageNotFoundError(name)

    monkeypatch.setattr(version_module, "version", missing_version)

    assert version_module._discover_version() == "0.0.0"
