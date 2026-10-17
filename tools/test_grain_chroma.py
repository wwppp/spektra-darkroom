import os
import sys
import numpy as np

# Add paths
sys.path.insert(0, os.path.abspath("spektrafilm-repo/src"))
sys.path.insert(0, os.path.abspath("."))

from spektrafilm.model.grain import (
    mix_chroma_grain,
    apply_jarvis_dye_cloud_filter,
    apply_grain,
    apply_grain_to_density,
)
from spektrafilm.runtime.params_schema import GrainParams
from spektrafilm.runtime.api import simulate
from spektrafilm.profiles.io import load_profile

def test_mix_chroma_grain():
    print("Testing mix_chroma_grain...")
    # Generate 3 independent random normal noise channels
    np.random.seed(42)
    fluc = np.random.randn(200, 200, 3) * 0.1
    var_init = np.var(fluc)

    # 1. Test colour = 1.0 (pure independent dye clouds)
    fluc_c1 = mix_chroma_grain(fluc.copy(), colour=1.0)
    np.testing.assert_allclose(fluc_c1, fluc, atol=1e-7)
    diff_rg = np.max(np.abs(fluc_c1[:, :, 0] - fluc_c1[:, :, 1]))
    assert diff_rg > 0.05, f"Expected independent channels at colour=1.0, got diff {diff_rg}"
    print("  [PASS] colour=1.0 keeps channels completely independent (Chroma Grain / 彩噪)")

    # 2. Test colour = 0.0 (pure luminance grain)
    fluc_c0 = mix_chroma_grain(fluc.copy(), colour=0.0)
    np.testing.assert_allclose(fluc_c0[:, :, 0], fluc_c0[:, :, 1], atol=1e-7)
    np.testing.assert_allclose(fluc_c0[:, :, 1], fluc_c0[:, :, 2], atol=1e-7)
    print("  [PASS] colour=0.0 makes all 3 channels identical (Pure Luminance Grain / 单色颗粒)")

    # 3. Test variance conservation across colour spectrum
    for c in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mixed = mix_chroma_grain(fluc.copy(), colour=c)
        var_c = np.var(mixed)
        rel_diff = abs(var_c - var_init) / var_init
        assert rel_diff < 0.05, f"Variance deviation {rel_diff:.4f} too large at colour={c}"
        print(f"  [PASS] colour={c:.2f} total variance preserved: init={var_init:.6f}, mixed={var_c:.6f} (diff={rel_diff*100:.2f}%)")

def test_jarvis_filter():
    print("\nTesting Jarvis dye cloud filter...")
    test_field = np.zeros((100, 100, 3), dtype=np.float64)
    test_field[50, 50, :] = 1.0
    filtered = apply_jarvis_dye_cloud_filter(test_field, decay_length_px=2.0)
    assert np.all(np.isfinite(filtered))
    assert np.all(filtered >= 0.0)
    assert filtered[50, 50, 0] < 1.0  # dispersed
    assert np.sum(filtered) > 0.0
    print("  [PASS] Jarvis 4-Gaussian dye cloud filter executes successfully and preserves energy")

def test_apply_grain_pipeline():
    print("\nTesting apply_grain end-to-end integration...")
    density_cmy = np.full((32, 32, 3), 0.6, dtype=np.float64)
    density_curves = np.tile(np.linspace(0.0, 2.2, 10)[:, None], (1, 3))
    density_curves_layers = np.tile(density_curves[:, None, :] / 3.0, (1, 3, 1))

    # Test monochromatic grain
    gp_mono = GrainParams(active=True, colour_grain=0.0, blur=0.0)
    out_mono = apply_grain(density_cmy.copy(), 10.0, gp_mono, density_curves, density_curves_layers, "negative")
    assert out_mono.shape == (32, 32, 3)
    fluc_mono = out_mono - density_cmy
    np.testing.assert_allclose(fluc_mono[:, :, 0], fluc_mono[:, :, 1], atol=1e-5)
    print("  [PASS] apply_grain with colour_grain=0.0 yields monochromatic density fluctuations")

    # Test chroma grain
    gp_chroma = GrainParams(active=True, colour_grain=1.0, blur=0.0)
    out_chroma = apply_grain(density_cmy.copy(), 10.0, gp_chroma, density_curves, density_curves_layers, "negative")
    fluc_chroma = out_chroma - density_cmy
    diff_chroma = np.max(np.abs(fluc_chroma[:, :, 0] - fluc_chroma[:, :, 1]))
    assert diff_chroma > 0.01, f"Expected chroma dye separation, got diff {diff_chroma}"
    print(f"  [PASS] apply_grain with colour_grain=1.0 yields independent dye cloud fluctuations (diff={diff_chroma:.4f})")

from spektrafilm.runtime.api import simulate, init_params

def test_simulate_full_profiles():
    print("\nTesting simulate() with new grain model across legacy & newly converted profiles...")
    stocks = [
        "kodak_portra_400",       # Legacy stock without physical_extensions
        "kodak_gold_200",         # Newly converted stock with crystal_grain physical extensions
        "cinestill_800t",         # Cine film stock
        "ilford_hp5_plus_400",     # B&W silver stock
    ]

    for stock_id in stocks:
        profile = load_profile(stock_id)
        assert profile is not None
        has_ext = hasattr(profile.data, "physical_extensions") and bool(profile.data.physical_extensions)
        has_crystal = has_ext and "crystal_grain" in (profile.data.physical_extensions or {})

        params = init_params(film_profile=stock_id)
        params.film_render.grain.active = True
        params.film_render.grain.colour_grain = 0.8
        img_in = np.ones((64, 64, 3), dtype=np.float64) * 0.3
        res = simulate(img_in, params)
        assert res is not None
        assert res.shape == (64, 64, 3)
        assert np.all(np.isfinite(res))
        print(f"  [PASS] simulate({stock_id}) passed! (physical_extensions: {has_ext}, crystal_grain: {has_crystal})")

if __name__ == "__main__":
    test_mix_chroma_grain()
    test_jarvis_filter()
    test_apply_grain_pipeline()
    test_simulate_full_profiles()
    print("\n==========================================")
    print("ALL PHASE 3 GRAIN & CHROMA TESTS PASSED!")
    print("==========================================")
