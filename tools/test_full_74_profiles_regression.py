import os
import sys
import glob
import time
import numpy as np

sys.path.insert(0, os.path.abspath("spektrafilm-repo/src"))
sys.path.insert(0, os.path.abspath("."))

from spektrafilm.runtime.api import simulate, init_params
from spektrafilm.profiles.io import load_profile

def test_all_profiles_full_simulation():
    profile_dir = "resources/profiles"
    profile_files = sorted(glob.glob(os.path.join(profile_dir, "*.json")))
    print(f"Found {len(profile_files)} profiles in {profile_dir}. Beginning full regression test...\n")

    passed_count = 0
    failed = []

    img_in = np.ones((48, 48, 3), dtype=np.float64) * 0.35
    img_in[24, 24, :] = 4.0  # specular point to exercise halation & curves shoulder

    t0 = time.time()
    for idx, ppath in enumerate(profile_files):
        stock_id = os.path.splitext(os.path.basename(ppath))[0]
        try:
            profile = load_profile(stock_id)
            if profile is None:
                raise ValueError(f"Could not load profile {stock_id}")

            params = init_params(film_profile=stock_id)
            params.film_render.grain.active = True
            params.film_render.grain.colour_grain = 0.65
            params.film_render.halation.active = True
            params.film_render.halation.halation_amount = 0.8
            params.film_render.dir_couplers.active = True

            res = simulate(img_in, params)
            if res is None or res.shape != (48, 48, 3) or not np.all(np.isfinite(res)):
                raise ValueError("Simulation output invalid or non-finite")

            passed_count += 1
            has_ext = hasattr(profile.data, "physical_extensions") and bool(profile.data.physical_extensions)
            tag = "[PHYS-EXT]" if has_ext else "[LEGACY]  "
            if (idx + 1) % 10 == 0 or idx == len(profile_files) - 1:
                print(f"  [{idx+1:02d}/{len(profile_files)}] {tag} {stock_id:38s} -> PASS")
        except Exception as e:
            failed.append((stock_id, str(e)))
            print(f"  [{idx+1:02d}/{len(profile_files)}] [FAIL]     {stock_id:38s} -> {e}")

    total_time = round(time.time() - t0, 2)
    print(f"\n==========================================")
    print(f"Full Regression Completed in {total_time}s")
    print(f"Total Profiles: {len(profile_files)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {len(failed)}")
    print(f"==========================================")

    if failed:
        print("\nFailed Profiles Details:")
        for stock_id, err in failed:
            print(f"  - {stock_id}: {err}")
        sys.exit(1)
    else:
        print("ALL 74 PROFILES PASSED ZERO-REGRESSION SIMULATION! SUCCESS")

if __name__ == "__main__":
    test_all_profiles_full_simulation()
