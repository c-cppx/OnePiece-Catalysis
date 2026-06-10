from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from onepiece.services import apply_dataset_query


def _welcome_app() -> None:
    from onepiece_studio.ui.welcome import run_welcome

    run_welcome()


def test_apply_dataset_query_handles_duplicate_index_labels() -> None:
    frame = pd.DataFrame(
        {"E": [1.0, 2.0, 3.0]},
        index=pd.Index(["CO@Cu", "CO@Cu", "O@Pt"], name="Name"),
    )
    row_keys = pd.Series(frame.index.astype(str), index=frame.index)

    # Without the duplicate-label guard this raised
    # "Item wrong length N instead of M".
    result = apply_dataset_query(frame, {}, row_key_series=row_keys, status_map={})

    assert len(result) == 3


def test_welcome_page_renders_tutorial_and_open_actions() -> None:
    app = AppTest.from_function(_welcome_app, default_timeout=120)
    app.run()

    assert not app.exception
    assert [t.value for t in app.title] == ["OnePiece Studio"]
    labels = [b.label for b in app.button]
    assert "Open the tutorial dataset" in labels
    assert "Open file" in labels


def test_welcome_tutorial_click_opens_workbench() -> None:
    app = AppTest.from_function(_welcome_app, default_timeout=120)
    app.run()

    tutorial_button = next(b for b in app.button if b.label == "Open the tutorial dataset")
    tutorial_button.click()
    app.run()

    assert not app.exception
    assert not app.error
    assert [t.value for t in app.title] == ["OnePiece Studio Tutorial Dataset"]
    assert "Controlroom" in [t.label for t in app.tabs]
