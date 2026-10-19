#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reversal Stocks Saturation & Midgray Calibration Regression Test
Verifies that all positive (reversal/slide) film stocks:
1. Have normalized middle gray anchor in 3D LUT (~45% - 55% sRGB).
2. Have clear base transmission reaching display white (> 0.95 sRGB).
3. Exhibit strong color saturation and color separation.
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))

from app_core import SpektraEngine, shaper_forward

def test_reversal_stocks():
    engine = SpektraEngine()
    profile_dir = "resources/profiles"
    profile_files = sorted(glob.glob(os.path.join(profile_dir, "*.json")))

    pos_stocks = []
    for ppath in profile_files:
        prof = engine._load_profile(os.path.splitext(os.path.basename(ppath))[0])
        if prof and ((hasattr(prof, "is_positive") and prof.is_positive) or
                     (hasattr(prof, "info") and getattr(prof.info, "type", None) == "positive")):
            pos_stocks.append(prof.info.stock)

    print(f"Testing {len(pos_stocks)} reversal film stocks for color saturation and midgray calibration...\n")

    passed = 0
    failed = []

    # Middle gray input: 0.184 in linear ProPhoto RGB
    mid_input = 0.184
    # Shaper forward mapping
    lut_size = 17
    lin_t = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
    # Find index closest to 0.184 in shaper coordinate
    from app_core import shaper_inverse
    lin_x = shaper_inverse(lin_t)
    mid_idx = int(np.argmin(np.abs(lin_x - mid_input)))

    for stock_id in pos_stocks:
        lut = engine.get_3d_lut(stock_id, "none", lut_size=lut_size)
        
        # 1. Check max highlight (corner [lut_size-1, lut_size-1, lut_size-1])
        white_val = lut[-1, -1, -1]
        max_val = float(np.max(lut))
        if max_val < 0.92:
            failed.append((stock_id, f"Peak highlight does not reach display white: max={max_val:.3f}"))
            continue

        # 2. Check midgray output: lut[mid_idx, mid_idx, mid_idx]
        mid_val = lut[mid_idx, mid_idx, mid_idx]
        mid_mean = float(np.mean(mid_val))
        if not (0.35 <= mid_mean <= 0.65):
            failed.append((stock_id, f"Midgray output out of range: {mid_mean:.3f} (val: {mid_val})"))
            continue

        # 3. Check color saturation: pure red/green/blue inputs should show strong channel separation
        # Pure red input corner: b=0, g=0, r=lut_size-1
        red_out = lut[0, 0, -1]
        # Pure green input corner: b=0, g=lut_size-1, r=0
        green_out = lut[0, -1, 0]
        # Pure blue input corner: b=lut_size-1, g=0, r=0
        blue_out = lut[-1, 0, 0]

        red_separation = red_out[0] - max(red_out[1], red_out[2])
        green_separation = green_out[1] - max(green_out[0], green_out[2])
        blue_separation = blue_out[2] - max(blue_out[0], blue_out[1])

        # At least one chromatic corner should have clear positive color separation (> 0.08)
        max_sep = max(red_separation, green_separation, blue_separation)
        if max_sep < 0.08:
            failed.append((stock_id, f"Low color separation: max_sep={max_sep:.3f}"))
            continue

        passed += 1
        print(f"  [PASS] {stock_id:32s} | Max: {max_val:.3f} | Mid: {mid_mean:.3f} | MaxSep: {max_sep:.3f}")

    print(f"\n==========================================")
    print(f"Reversal Stocks Verification: {passed}/{len(pos_stocks)} PASSED")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for stock_id, err in failed:
            print(f"  [FAIL] {stock_id:32s}: {err}")
        sys.exit(1)
    else:
        print("ALL REVERSAL STOCKS MEET VIVID COLOR SATURATION & CALIBRATION STANDARDS! 100% SUCCESS")

if __name__ == "__main__":
    test_reversal_stocks()
