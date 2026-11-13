"""UI smoke test: every page of the Streamlit app renders without exceptions (headless AppTest)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["home", "synthetic_cell", "channel_1", "channel_2", "channel_3", "comparison", "validation_page"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.run()
    at.switch_page(f"app_pages/{page}.py")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
