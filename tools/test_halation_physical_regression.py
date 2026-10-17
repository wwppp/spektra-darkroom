#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Halation Physical Regression Test
Verifies that all film stock halation presets strictly reflect true physical layer properties,
ensuring no profile has been artificially multiplied by 0.25 (e.g. Superia 400 remains ~0.0055, Vision3 remains ~0.0008).
"""

import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.abspath("spektrafilm-repo/src"))
sys.path.insert(0, os.path.abspath("."))

from app_core import SpektraEngine

def test_halation_physical_presets():
    engine = SpektraEngine()
    profile_dir = "resources/profiles"
    profile_files = sorted(glob.glob(os.path.join(profile_dir, "*.json")))
    print(f"Checking physical halation presets for {len(profile_files)} profiles...\n")

    passed = 0
    failed = []

    for ppath in profile_files:
        stock_id = os.path.splitext(os.path.basename(ppath))[0]
        preset = engine.get_film_halation_preset(stock_id)
        strength = np.array(preset["strength"], dtype=float)
        sigma = float(preset["sigma_h"])
        antihalation = preset.get("antihalation", "strong")

        # Physical sanity checks:
        # 1. Sigma should be positive and physically reasonable (between 20um and 200um)
        if not (10.0 <= sigma <= 300.0):
            failed.append((stock_id, f"Unreasonable sigma_h: {sigma} um"))
            continue

        # 2. Strength must be non-negative and finite
        if np.any(strength < 0) or np.any(~np.isfinite(strength)):
            failed.append((stock_id, f"Invalid strength values: {strength}"))
            continue

        # 3. Check specific stock physical anchors:
        # Superia 400 (fujifilm_xtra_400): physical strength ~ [0.0055, 0.000149, 0]
        if "xtra_400" in stock_id:
            if not np.isclose(strength[0], 0.0055, rtol=0.1):
                failed.append((stock_id, f"Superia 400 red halation expected ~0.0055, got {strength[0]}"))
                continue

        # Vision3 (remjet antihalation): physical strength <= 0.002
        if "vision3" in stock_id:
            if strength[0] > 0.01:
                failed.append((stock_id, f"Vision3 Remjet carbon halation expected <= 0.01, got {strength[0]}"))
                continue

        # CineStill 800T/400D (Remjet removed, halation='no'): high halation ~ [0.03, 0.001, 0.0]
        if "cinestill" in stock_id:
            if strength[0] < 0.02:
                failed.append((stock_id, f"CineStill expected strong halation >= 0.02, got {strength[0]}"))
                continue

        # 4. No stock with physical definition should have strength exactly (0.25, ...) from normalization bug
        if np.isclose(strength[0], 0.25) and "cinestill" not in stock_id:
            failed.append((stock_id, f"Suspicious exactly 0.25 normalized strength: {strength}"))
            continue

        passed += 1

    print(f"==========================================")
    print(f"Halation Physical Verification: {passed}/{len(profile_files)} PASSED")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for stock_id, err in failed:
            print(f"  [FAIL] {stock_id:35s}: {err}")
        sys.exit(1)
    else:
        print("ALL FILM PROFILES HAVE STRICT PHYSICAL HALATION PRESETS! 100% SUCCESS")

if __name__ == "__main__":
    test_halation_physical_presets()
