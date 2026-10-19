"""
Test Pre-LUT Grain and Highlight Behavior Refactoring:
1. Verify get_film_grain_preset loads accurate crystal_grain metadata across all profile types.
2. Verify Pre-LUT grain physics in highlights (S-curve compression, zero pepper specks, zero plastic clipping).
3. Verify AppCore.export_image CPU pipeline with Pre-LUT grain.
"""
import os
import sys
import numpy as np

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, root_dir)

from app_core import SpektraEngine

def test_grain_presets():
    print("Testing get_film_grain_preset...")
    core = SpektraEngine()

    # 1. Color negative film (Gold 200)
    p_gold = core.get_film_grain_preset("kodak_gold_200")
    assert abs(p_gold["size_mm"] - 0.005) < 1e-4, f"Gold 200 size_mm mismatch: {p_gold['size_mm']}"
    assert abs(p_gold["strength"] - 0.010) < 1e-3, f"Gold 200 strength mismatch: {p_gold['strength']}"
    assert len(p_gold["layer_weights"]) == 3, "Gold 200 layer_weights length != 3"
    assert p_gold["layer_weights"][2] > p_gold["layer_weights"][0], "Blue layer should be coarsest in color negative"
    print(f"  [PASS] kodak_gold_200 -> size={p_gold['size_mm']}, strength={p_gold['strength']}, weights={p_gold['layer_weights']}")

    # 2. Tungsten film (CineStill 800T)
    p_800t = core.get_film_grain_preset("cinestill_800t")
    assert abs(p_800t["layer_weights"][2] - 2.71) < 0.05, f"800T blue layer weight should be ~2.71, got {p_800t['layer_weights'][2]}"
    print(f"  [PASS] cinestill_800t -> size={p_800t['size_mm']}, strength={p_800t['strength']}, weights={p_800t['layer_weights']}")

    # 3. B&W film (Tri-X 400)
    p_trix = core.get_film_grain_preset("kodak_tri_x_400")
    assert abs(p_trix["size_mm"] - 0.008) < 1e-4, f"Tri-X size_mm mismatch: {p_trix['size_mm']}"
    assert abs(p_trix["strength"] - 0.017) < 1e-4, f"Tri-X strength mismatch: {p_trix['strength']}"
    assert p_trix["layer_weights"] == [1.0, 1.0, 1.0], f"Tri-X layer weights should be [1,1,1], got {p_trix['layer_weights']}"
    print(f"  [PASS] kodak_tri_x_400 -> size={p_trix['size_mm']}, strength={p_trix['strength']}, weights={p_trix['layer_weights']}")

    # 4. Fine-grain film (Ektar 100)
    p_ektar = core.get_film_grain_preset("kodak_ektar_100")
    assert p_ektar["strength"] < p_gold["strength"], "Ektar 100 should have less grain than Gold 200"
    print(f"  [PASS] kodak_ektar_100 -> size={p_ektar['size_mm']}, strength={p_ektar['strength']}, weights={p_ektar['layer_weights']}")

    # 5. Legacy profile fallback (kodak_portra_400)
    p_portra = core.get_film_grain_preset("kodak_portra_400")
    assert p_portra["size_mm"] > 0 and p_portra["strength"] > 0
    print(f"  [PASS] kodak_portra_400 (legacy fallback) -> size={p_portra['size_mm']}, strength={p_portra['strength']}")

def test_highlight_compression():
    print("\nTesting Highlight Grain Compression & Pepper Specks Prevention...")
    core = SpektraEngine()

    # Get LUT for Gold 200 + Kodak Endura
    lut = core.get_3d_lut("kodak_gold_200", "kodak_portra_endura", lut_size=33)
    lut_size = lut.shape[0]

    # Create synthetic test patches:
    # 1. Pure specular highlight (linear = 3.0)
    # 2. White furry hat highlight (linear = 1.3)
    # 3. 18% gray midtone (linear = 0.18)
    # 4. Shadow (linear = 0.02)
    def shaper_forward(lin):
        SHAPER_C = 60.0
        SHAPER_INV_LOG = 0.16192058
        return np.clip(np.log(1.0 + np.maximum(0.0, lin) * SHAPER_C) * SHAPER_INV_LOG, 0.0, 1.0)

    def sample_lut(lin_patch):
        coords = np.clip(shaper_forward(lin_patch) * (lut_size - 1), 0.0, lut_size - 1.0001)
        idx0 = coords.astype(np.int32)
        idx1 = np.minimum(idx0 + 1, lut_size - 1)
        d = coords - idx0
        r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
        r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]
        c000 = lut[b0, g0, r0]
        c100 = lut[b0, g0, r1]
        c010 = lut[b0, g1, r0]
        c110 = lut[b0, g1, r1]
        c001 = lut[b1, g0, r0]
        c101 = lut[b1, g0, r1]
        c011 = lut[b1, g1, r0]
        c111 = lut[b1, g1, r1]
        dr, dg, db = d[..., 0:1], d[..., 1:2], d[..., 2:3]
        c00 = c000 * (1.0 - dr) + c100 * dr
        c01 = c001 * (1.0 - dr) + c101 * dr
        c10 = c010 * (1.0 - dr) + c110 * dr
        c11 = c011 * (1.0 - dr) + c111 * dr
        c0 = c00 * (1.0 - dg) + c10 * dg
        c1 = c01 * (1.0 - dg) + c11 * dg
        return c0 * (1.0 - db) + c1 * db

    # Simulate 10,000 pixels of a white furry hat patch (linear = 1.3)
    np.random.seed(42)
    N = 10000
    hat_linear = np.ones((N, 1, 3), dtype=np.float32) * 1.3

    # Apply Pre-LUT grain
    eff_strength = 1.0
    lin_luma = 0.2126 * hat_linear[..., 0] + 0.7152 * hat_linear[..., 1] + 0.0722 * hat_linear[..., 2]
    dens_mod = (1.0 - np.exp(-lin_luma * 15.0)) / (1.0 + (lin_luma / 0.28) ** 2.2)
    noise = np.random.randn(N, 1, 3).astype(np.float32) * 0.5 * np.array([0.7, 1.0, 1.35], dtype=np.float32)
    dD = noise * (eff_strength * 0.35 * dens_mod[..., None])
    hat_linear_grained = np.maximum(0.0, hat_linear * np.exp(-dD * 2.3025851))

    # Output through 3D LUT
    res_base = sample_lut(hat_linear)
    res_grained = sample_lut(hat_linear_grained)

    # 1. Verify NO clipping at 1.0 (paper base naturally around 0.95 - 0.98)
    assert np.all(res_grained <= 1.0), "Values exceeded 1.0!"
    print(f"  [PASS] White Hat Base Color: R={res_base[0,0,0]:.3f}, G={res_base[0,0,1]:.3f}, B={res_base[0,0,2]:.3f}")
    print(f"  [PASS] White Hat Grained Range: R=[{res_grained[...,0].min():.3f}, {res_grained[...,0].max():.3f}], G=[{res_grained[...,1].min():.3f}, {res_grained[...,1].max():.3f}]")

    # 2. Verify NO dark pepper specks (min values don't collapse, preserving creamy soft paper texture)
    min_val = np.min(res_grained)
    base_min = np.min(res_base)
    ratio = min_val / max(1e-4, base_min)
    assert ratio > 0.80, f"Dark pepper speck detected! Min/Base ratio was {ratio:.3f}"
    print(f"  [PASS] Minimum pixel brightness in highlight preserves {ratio*100:.1f}% of base (Zero pepper specks, pure creamy tone)")

    # 3. Check grain modulation ratio between highlight and midtone
    mid_linear = np.ones((N, 1, 3), dtype=np.float32) * 0.18
    mid_luma = 0.18
    dens_mod_m = (1.0 - np.exp(-mid_luma * 15.0)) / (1.0 + (mid_luma / 0.28) ** 2.2)
    dD_m = noise * (eff_strength * 0.35 * dens_mod_m)
    mid_linear_grained = np.maximum(0.0, mid_linear * np.exp(-dD_m * 2.3025851))
    res_mid_grained = sample_lut(mid_linear_grained)

    std_hl = float(np.mean(np.std(res_grained, axis=0)))
    std_mid = float(np.mean(np.std(res_mid_grained, axis=0)))
    print(f"  [PASS] Per-channel Noise RMS: Midtone={std_mid*255.0:.2f} levels, Highlight={std_hl*255.0:.2f} levels (Ratio={std_hl/std_mid:.3f})")
    assert std_mid > std_hl * 2.5, "Midtone grain should naturally be at least 2.5x more pronounced than highlight grain!"
    print("  [PASS] Paper S-curve shoulder naturally rolls off highlight grain amplitude!")

def test_cpu_export():
    print("\nTesting AppCore CPU Export Pipeline with Pre-LUT Grain...")
    core = SpektraEngine()

    # Synthetic image 512x512
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Gradient from shadow to highlight
    for y in range(h):
        val = (y / float(h)) ** 1.8 * 2.5
        img[y, :, :] = val

    out_path = os.path.abspath("test_pre_lut_export.jpg")
    if os.path.exists(out_path):
        os.remove(out_path)

    params = {
        "engine": "wysiwyg",
        "film_stock": "kodak_gold_200",
        "paper_stock": "kodak_portra_endura",
        "grain": 0.5,
        "format": "jpeg",
        "quality": 95,
        "base_ev": 0.0
    }

    core.raw_preview = img
    res = core.export_image(params, out_path, format_type="jpeg")
    assert res.get("success"), f"Export failed: {res}"
    assert os.path.exists(out_path), "Export file not written"
    fsize = os.path.getsize(out_path)
    assert fsize > 1000, f"Export file too small: {fsize} bytes"
    print(f"  [PASS] CPU Export Succeeded! Size={fsize} bytes, Path={out_path}")

    # Clean up test file
    if os.path.exists(out_path):
        os.remove(out_path)

if __name__ == "__main__":
    test_grain_presets()
    test_highlight_compression()
    test_cpu_export()
    print("\n==========================================")
    print("ALL PRE-LUT GRAIN & EXPORT TESTS PASSED!")
    print("==========================================")
