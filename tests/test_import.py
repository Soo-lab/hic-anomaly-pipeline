import importlib


def test_package_imports_and_version():
    pkg = importlib.import_module("hic_anomaly")
    assert isinstance(pkg.__version__, str) and pkg.__version__


def test_public_modules_import():
    for name in ("io_utils", "matrix", "detect", "viz", "crossref", "cli"):
        importlib.import_module(f"hic_anomaly.{name}")


def test_cli_entry_points_callable():
    from hic_anomaly.cli import main as cli_main
    from hic_anomaly.crossref import main as crossref_main

    assert callable(cli_main)
    assert callable(crossref_main)


def test_root_shims_reexport():
    import cross_refrence_hits
    import universal_hic_pipeline

    assert callable(universal_hic_pipeline.main)
    assert callable(universal_hic_pipeline.run_ai)
    assert callable(cross_refrence_hits.cross_reference)
