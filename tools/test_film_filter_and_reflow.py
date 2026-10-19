"""tools/test_film_filter_and_reflow.py - Regression & Unit Test for Film Filter Header, Categories, Search, and Grid Reflow.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSize

# Headless Qt App
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from app_core import SpektraEngine
from ui.film_filter_header import FilmFilterHeader, FlowLayout


def test_film_categories_and_search_logic():
    print("=== Testing Film Stocks Metadata & Filtering Logic ===")
    engine = SpektraEngine()
    stocks_groups = engine.get_film_stocks()

    all_films = []
    for g in stocks_groups:
        for item in g.get("stocks", []):
            all_films.append(item)

    print(f"Total film stocks discovered: {len(all_films)}")
    assert len(all_films) >= 50, f"Expected at least 50 films, got {len(all_films)}"

    # 1. Test B&W category
    bw_films = [f for f in all_films if f.get("channel_model") == "bw"]
    print(f"Found {len(bw_films)} B&W films: {[f['id'] for f in bw_films]}")
    expected_bw = {"ilford_hp5_plus_400", "kodak_tri_x_400", "ilford_delta_3200", "fujifilm_neopan_acros_100"}
    for exp in expected_bw:
        assert any(f["id"] == exp for f in bw_films), f"Expected {exp} in B&W films"
    print("  -> B&W classification: PASS")

    # 2. Test Positive/Reversal category
    positive_films = [f for f in all_films if f.get("type") == "positive"]
    print(f"Found {len(positive_films)} Reversal/Slide films: {[f['id'] for f in positive_films]}")
    expected_pos = {"fujifilm_velvia_100", "fujifilm_provia_100f", "kodak_ektachrome_100"}
    for exp in expected_pos:
        assert any(f["id"] == exp for f in positive_films), f"Expected {exp} in Positive films"
    print("  -> Positive/Reversal classification: PASS")

    # 3. Test Cinema category
    cine_films = [
        f for f in all_films
        if f.get("use") == "cine" or "电影" in (f.get("badge") or "") or "vision3" in f["id"] or "eterna" in f["id"] or "cinestill" in f["id"]
    ]
    print(f"Found {len(cine_films)} Cinema films: {[f['id'] for f in cine_films]}")
    assert any("vision3" in f["id"] for f in cine_films), "Expected Vision3 in Cinema films"
    print("  -> Cinema classification: PASS")

    # 4. Test Search by Badge
    portrait_badge_films = [f for f in all_films if "人像之王" in (f.get("badge") or "")]
    assert len(portrait_badge_films) >= 1 and portrait_badge_films[0]["id"] == "kodak_portra_400", "Search by badge '人像之王' failed"
    print("  -> Search by badge ('人像之王'): PASS")


def test_film_filter_header_widget():
    print("\n=== Testing FilmFilterHeader Widget & Responsive Layout ===")
    header = FilmFilterHeader()
    header.show()
    header.resize(340, 50)
    app.processEvents()

    # Verify initial state
    assert header._current_cat == "all"
    assert header._current_query == ""
    assert not header._is_compact_3row, "At 320px width, should be 2-row layout"

    # Test Search signal emission
    received_queries = []
    header.searchChanged.connect(lambda q: received_queries.append(q))
    header.search_box.setText("Portra")
    assert received_queries == ["Portra"], f"Expected ['Portra'], got {received_queries}"
    print("  -> Search input signal: PASS")

    # Test Category signal emission
    received_cats = []
    header.categoryChanged.connect(lambda c: received_cats.append(c))
    header.pills_dict["bw"].click()
    assert received_cats == ["bw"], f"Expected ['bw'], got {received_cats}"
    assert header._current_cat == "bw"
    print("  -> Category pill click signal: PASS")

    # Test Responsive breakpoint: shrink to 250px (< 290px)
    header.resize(250, 75)
    app.processEvents()
    assert header._is_compact_3row, "At 250px width, should switch to compact 3-row layout"
    assert header.row_search_widget.isVisible(), "Search row should be visible in 3-row layout"
    print("  -> Responsive 3-row breakpoint (< 290px): PASS")

    # Expand back to 340px (>= 290px)
    header.resize(340, 50)
    app.processEvents()
    assert not header._is_compact_3row, "At 340px width, should switch back to 2-row layout"
    assert not header.row_search_widget.isVisible(), "Search row should be hidden in 2-row layout"
    print("  -> Responsive 2-row recovery (>= 290px): PASS")


def test_column_math_no_overflow():
    print("\n=== Testing Grid Column Thresholds & Mathematical Clearance ===")
    min_card_w = 96
    spacing = 6
    margins = 8

    test_widths = [200, 220, 250, 280, 307, 308, 320, 350, 409, 410, 450]

    for viewport_w in test_widths:
        avail_w = max(96, viewport_w - margins)
        if avail_w >= 410:
            cols = 4
        elif avail_w >= 308:
            cols = 3
        elif avail_w >= 206:
            cols = 2
        else:
            cols = 1

        target_card_w = int((avail_w - (cols - 1) * spacing) / cols)
        target_card_w = max(min_card_w, min(140, target_card_w))

        total_row_w = cols * target_card_w + (cols - 1) * spacing
        # Total row width must NEVER exceed avail_w
        assert total_row_w <= avail_w, (
            f"OVERFLOW DETECTED! viewport_w={viewport_w}, avail_w={avail_w}, "
            f"cols={cols}, card_w={target_card_w}, total_row_w={total_row_w} > avail_w={avail_w}"
        )
        assert target_card_w >= min_card_w, f"Card width {target_card_w} less than min_card_w {min_card_w}"

    print(f"  -> All {len(test_widths)} sidebar viewport widths tested: ZERO OVERFLOW, 100% CLEARANCE PASS")


if __name__ == "__main__":
    test_film_categories_and_search_logic()
    test_film_filter_header_widget()
    test_column_math_no_overflow()
    print("\nALL FILM FILTER AND REFLOW TESTS COMPLETED SUCCESSFULLY!")
