import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'spektrafilm-repo', 'src'))
import numpy as np
from app_core import SpektraEngine

core = SpektraEngine()
stocks_to_test = [
    # Color Negative
    'kodak_kodacolor_100',
    'kodak_ektar_100',
    'kodak_gold_200',
    'kodak_portra_400',
    'fujifilm_c200',
    'kodak_ultramax_400',
    'cinestill_400d',
    'cinestill_800t',
    'fujifilm_pro_400h',
    'kodak_vision3_250d',
    # Reversal / Slide
    'agfa_agfachrome_ct_100',
    'fujifilm_velvia_50',
    'kodak_ektachrome_100',
    'fujifilm_provia_100f',
    'kodak_kodachrome_64',
    'fujifilm_astia_100f',
    # B&W Monochrome
    'ilford_hp5_plus_400',
    'kodak_tmax_100',
    'kodak_tri_x_400',
    'fujifilm_neopan_acros_100'
]

print(f"{'Stock':<32} | {'Center (8,8,8)':<26} | {'Mean':<6} | {'Max':<6} | {'Status':<8}")
print('-' * 85)

for s in stocks_to_test:
    lut = core.get_3d_lut(s, 'kodak_endura_premier', lut_size=17)
    center = lut[8, 8, 8].round(3)
    c_str = f"[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"
    mean_val = lut.mean()
    max_val = lut.max()
    status = 'OK' if (mean_val > 0.30 and max_val > 0.8 and center.mean() > 0.40) else 'FAIL'
    print(f"{s:<32} | {c_str:<26} | {mean_val:.3f}  | {max_val:.3f}  | {status:<8}")
