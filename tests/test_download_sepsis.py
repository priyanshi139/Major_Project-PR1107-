import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "download_sepsis", Path(__file__).resolve().parents[1] / "scripts" / "download_sepsis.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)


def test_parse_listing_extracts_psv_only():
    html = ('<a href="../">Parent</a><a href="p000001.psv">x</a>'
            '<a href="p000002.psv">y</a><a href="p000001.psv">dup</a><a href="notes.txt">n</a>')
    assert ds.parse_listing(html) == ["p000001.psv", "p000002.psv"]


def test_parse_listing_empty():
    assert ds.parse_listing("<html></html>") == []
