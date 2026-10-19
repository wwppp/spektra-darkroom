import os
import sys
import numpy as np
import rawpy
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))

from spektrafilm.runtime.api import simulate, init_params
from spektrafilm.profiles.io import load_profile
from app_core import SpektraEngine

dng_path = os.path.abspath("测试底片/IMG_8425.DNG")
print("Reading DNG:", dng_path)

with rawpy.imread(dng_path) as raw:
    # Match our engine's raw decode settings
    raw_rgb = raw.postprocess(
        use_camera_wb=True,
        half_size=True,
        no_auto_bright=True,
        output_color=rawpy.ColorSpace.sRGB,
        output_bps=16
    ).astype(np.float64) / 65535.0

print(f"Decoded RAW RGB shape: {raw_rgb.shape}, mean: {raw_rgb.mean():.4f}, max: {raw_rgb.max():.4f}")

# Save decoded linear/sRGB raw preview
Image.fromarray((np.clip(raw_rgb, 0, 1) * 255).astype(np.uint8)).save("scratch_raw_linear.jpg", quality=90)

core = SpektraEngine()
# Test 1: SpektraEngine 3D LUT on Velvia 50
print("\n--- Testing SpektraEngine 3D LUT on Velvia 50 ---")
lut_v50 = core.get_3d_lut("fujifilm_velvia_50", "none", lut_size=33)
print("LUT shape:", lut_v50.shape, "min:", lut_v50.min(), "max:", lut_v50.max(), "mean:", lut_v50.mean())

# Test 2: SpektraEngine 3D LUT on Velvia 100
print("\n--- Testing SpektraEngine 3D LUT on Velvia 100 ---")
lut_v100 = core.get_3d_lut("fujifilm_velvia_100", "none", lut_size=33)
print("LUT shape:", lut_v100.shape, "min:", lut_v100.min(), "max:", lut_v100.max(), "mean:", lut_v100.mean())

# Test 3: Raw spektrafilm simulate() on Velvia 50
print("\n--- Testing Native spektrafilm.simulate() on Velvia 50 ---")
params = init_params(film_profile="fujifilm_velvia_50")
params.io.scan_film = True
params.print = None
params.settings.neutral_print_filters_from_database = False
params.film_render.grain.active = False
params.film_render.halation.active = False
params.film_render.dir_couplers.active = False
res_v50 = simulate(raw_rgb, params)
print("simulate res_v50 shape:", res_v50.shape, "min:", res_v50.min(), "max:", res_v50.max(), "mean:", res_v50.mean())
Image.fromarray((np.clip(res_v50, 0, 1) * 255).astype(np.uint8)).save("scratch_spektrafilm_v50.jpg", quality=90)

# Test 4: What if scan_film is False and printed on a paper? Or what did our software do?
# In our software: what was passed into get_3d_lut?
# Let's check what the user's SDC had:
# "paper_profile": "none", "print_mode": "scan", "exposure_ev": 1.25
params_dict = {
    "exposure_ev": 1.25,
    "highlights": 0.0,
    "shadows": 0.0,
    "color_temp": 5925.0,
    "tint": 1.0,
    "base_ev": 1.25
}
# Apply our software offline render
render_res = core.render_offscreen(raw_rgb.astype(np.float32), "fujifilm_velvia_50", "none", params_dict)
Image.fromarray((np.clip(render_res, 0, 1) * 255).astype(np.uint8)).save("scratch_spektra_app_v50.jpg", quality=90)
print("Saved scratch_spektra_app_v50.jpg")
