import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("spektrafilm-repo/src"))
sys.path.insert(0, os.path.abspath("."))

from spektrafilm.model.diffusion import apply_halation_um
from spektrafilm.runtime.params_schema import HalationParams
from spektrafilm.runtime.api import simulate, init_params
from spektrafilm.profiles.io import load_profile

def test_snell_tir_dispersion():
    print("Testing Snell TIR halation & chromatic dispersion...")
    film_base = {
        "thickness_mm": 0.13,
        "refractive_index": 1.48,
        "record_refractive_indices": [1.474, 1.480, 1.489],
        "antihalation": "no"
    }

    hp = HalationParams(
        active=True,
        halation_amount=1.0,
        enable_snell_tir=True,
        film_base=film_base
    )

    # Synthetic point light source image (single white specular highlight in dark background)
    raw = np.zeros((120, 120, 3), dtype=np.float64)
    raw[60, 60, :] = 5.0  # bright highlight

    pixel_size_um = 10.0  # 10um per pixel
    out = apply_halation_um(raw.copy(), hp, pixel_size_um)

    # Check finite and positive
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)

    # At distance from center (e.g. at radius 18 pixels = 180um),
    # check that Red halation energy is strictly greater than Green and Blue due to dispersion & no-antihalation
    halo_r = out[60, 78, 0]
    halo_g = out[60, 78, 1]
    halo_b = out[60, 78, 2]

    assert halo_r > halo_g > halo_b, f"Expected red-dominant halation ring, got R={halo_r}, G={halo_g}, B={halo_b}"
    print(f"  [PASS] Red halation ({halo_r:.4f}) > Green ({halo_g:.4f}) > Blue ({halo_b:.4f}) as physically predicted by Snell TIR!")

def test_snell_halation_profiles():
    print("\nTesting halation end-to-end with simulate() across key profiles...")
    stocks = [
        "cinestill_800t",         # no remjet, prominent red halation
        "kodak_portra_400",       # standard still negative
        "kodak_vision3_500t_fotu",# cine stock with film_base
        "harman_phoenix_200",     # experimental no-antihalation stock
    ]

    for stock_id in stocks:
        profile = load_profile(stock_id)
        assert profile is not None
        has_ext = hasattr(profile.data, "physical_extensions") and bool(profile.data.physical_extensions)
        has_fb = has_ext and bool((profile.data.physical_extensions or {}).get("halation") or (profile.data.physical_extensions or {}).get("film_base"))

        params = init_params(film_profile=stock_id)
        params.film_render.halation.active = True
        params.film_render.halation.halation_amount = 1.0
        params.film_render.halation.boost_ev = 1.0

        # Image with bright highlight
        img_in = np.zeros((64, 64, 3), dtype=np.float64)
        img_in[32, 32, :] = 8.0

        res = simulate(img_in, params)
        assert res is not None
        assert res.shape == (64, 64, 3)
        assert np.all(np.isfinite(res))
        print(f"  [PASS] simulate({stock_id}) halation passed! (film_base: {has_fb})")

if __name__ == "__main__":
    test_snell_tir_dispersion()
    test_snell_halation_profiles()
    print("\n==========================================")
    print("ALL PHASE 4 SNELL HALATION TESTS PASSED! SUCCESS")
    print("==========================================")
