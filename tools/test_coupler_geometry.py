import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))

from spektrafilm.model.couplers import compute_dir_couplers_matrix
from spektrafilm.runtime.params_schema import DirCouplersParams
from spektrafilm.runtime.api import simulate, init_params
from spektrafilm.profiles.io import load_profile

def test_coupler_geometry_derivation():
    print("Testing CouplerGeometry dynamic matrix derivation...")
    cg_sample = {
        "interlayerTransmission": [0.4898, 0.5644],
        "layerDepthUM": [14.0, 8.0, 3.0],
        "release": [1.5286, 0.1332, 0.6576],
        "selfRetention": 0.2248
    }

    # Base params
    cp = DirCouplersParams(
        active=True,
        inhibition_samelayer=1.0,
        inhibition_interlayer=1.0,
        coupler_geometry={"coupler_geometry": cg_sample}
    )

    M_base = compute_dir_couplers_matrix(cp)
    assert M_base.shape == (3, 3)
    assert np.all(np.isfinite(M_base))
    assert np.all(M_base >= 0.0)
    print("  [PASS] CouplerGeometry derives a valid non-negative 3x3 DIR matrix")

    # Test increasing interlayer reach
    cp_high_reach = DirCouplersParams(
        active=True,
        inhibition_samelayer=1.0,
        inhibition_interlayer=2.0,
        coupler_geometry={"coupler_geometry": cg_sample}
    )
    M_high_reach = compute_dir_couplers_matrix(cp_high_reach)

    # Diagonal should remain equal, off-diagonals should increase
    np.testing.assert_allclose(np.diag(M_base), np.diag(M_high_reach), atol=1e-6)
    assert M_high_reach[0, 1] > M_base[0, 1]
    assert M_high_reach[1, 2] > M_base[1, 2]
    print(f"  [PASS] Increasing interlayer reach increases off-diagonal penetration ({M_base[0, 1]:.4f} -> {M_high_reach[0, 1]:.4f})")

    # Test increasing same-layer self-retention
    cp_high_self = DirCouplersParams(
        active=True,
        inhibition_samelayer=1.5,
        inhibition_interlayer=1.0,
        coupler_geometry={"coupler_geometry": cg_sample}
    )
    M_high_self = compute_dir_couplers_matrix(cp_high_self)
    assert M_high_self[0, 0] > M_base[0, 0]
    print(f"  [PASS] Increasing samelayer increases diagonal self-retention ({M_base[0, 0]:.4f} -> {M_high_self[0, 0]:.4f})")

def test_couplers_simulate_profiles():
    print("\nTesting Couplers simulate() end-to-end...")
    stocks = [
        "cinestill_800t",
        "kodak_portra_400",
        "kodak_gold_200",
        "fujifilm_provia_100f_fotu",
    ]

    for stock_id in stocks:
        profile = load_profile(stock_id)
        assert profile is not None
        has_ext = hasattr(profile.data, "physical_extensions") and bool(profile.data.physical_extensions)
        has_cg = has_ext and "couplers" in (profile.data.physical_extensions or {})

        params = init_params(film_profile=stock_id)
        params.film_render.dir_couplers.active = True
        params.film_render.dir_couplers.inhibition_interlayer = 1.2
        params.film_render.dir_couplers.inhibition_samelayer = 1.1

        img_in = np.ones((64, 64, 3), dtype=np.float64) * 0.4
        res = simulate(img_in, params)
        assert res is not None
        assert res.shape == (64, 64, 3)
        assert np.all(np.isfinite(res))
        print(f"  [PASS] simulate({stock_id}) couplers passed! (couplers: {has_cg})")

if __name__ == "__main__":
    test_coupler_geometry_derivation()
    test_couplers_simulate_profiles()
    print("\n==========================================")
    print("ALL PHASE 5 COUPLER GEOMETRY TESTS PASSED! SUCCESS")
    print("==========================================")
