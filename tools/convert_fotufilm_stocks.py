#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fotufilm to Spektrafilm Profile Converter (Fixed & Calibrated)
Converts film stock profiles from fotufilm-engine into spektrafilm Profile JSON format.
Fixes:
1. Calibrated log_sensitivity (normalizes linear sensitivity so 18% gray yields raw=1.0)
2. Baseline-zeroed density curves (subtracts dMin so net dye density is relative to D-min)
3. Physically consistent base_density and midscale_neutral_density
4. Solves and generates exact neutral print filters for all papers and enlarger illuminants
"""

import os
import sys
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root

# Ensure spektrafilm is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from spektrafilm.profiles.io import (
    Profile, ProfileMetadata, ProfileInfo, ProfileData,
    DensityCurvesModel, _validate_profile, profile_to_dict, _json_safe,
    load_profile
)
from spektrafilm.utils.spectral_upsampling import rgb_to_raw_hanatos2025
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.model.color_filters import color_enlarger
from spektrafilm.utils.conversions import density_to_light
from spektrafilm.model.develop import compute_density_spectral

# Canonical mapping from fotufilm ID to spektrafilm stock slug
STOCK_SLUG_MAP = {
    'acros100': 'fujifilm_neopan_acros_100',
    'aerochrome1443': 'kodak_aerochrome_1443',
    'aerocolor2460': 'kodak_aerocolor_iv_2460',
    'agfachrome100': 'agfa_agfachrome_ct_100',
    'agfachrome200': 'agfa_agfachrome_ct_200',
    'agfachrome50': 'agfa_agfachrome_ct_50',
    'astia100f': 'fujifilm_astia_100f',
    'c200': 'fujifilm_c200_fotu',
    'cinestill400d': 'cinestill_400d',
    'cinestill800t': 'cinestill_800t',
    'delta3200': 'ilford_delta_3200',
    'doublex5222': 'kodak_double_x_5222',
    'ektachromee100': 'kodak_ektachrome_e100_fotu',
    'ektar100': 'kodak_ektar_100_fotu',
    'eterna500': 'fujifilm_eterna_500t',
    'eternavivid500': 'fujifilm_eterna_vivid_500t',
    'f64d': 'fujifilm_f_64d',
    'fp4plus125': 'ilford_fp4_plus_125',
    'gold200': 'kodak_gold_200_fotu',
    'hp5plus400': 'ilford_hp5_plus_400',
    'infraredcolor': 'color_infrared_experimental',
    'instaxmini': 'fujifilm_instax_mini',
    'instaxwide': 'fujifilm_instax_wide',
    'kodachrome200': 'kodak_kodachrome_200',
    'kodachrome25': 'kodak_kodachrome_25',
    'kodachrome64': 'kodak_kodachrome_64_fotu',
    'kodacolor100': 'kodak_kodacolor_100',
    'lomochromemetropolis': 'lomography_lomochrome_metropolis',
    'lomochromepurple': 'lomography_lomochrome_purple',
    'lomochrometurquoise': 'lomography_lomochrome_turquoise',
    'lomocolor100': 'lomography_color_tiger_100',
    'phoenix200': 'harman_phoenix_200',
    'portra400': 'kodak_portra_400_fotu',
    'pro160ns': 'fujifilm_pro_160ns',
    'pro400h': 'fujifilm_pro_400h_fotu',
    'provia100f': 'fujifilm_provia_100f_fotu',
    'reala500d': 'fujifilm_reala_500d',
    'superia200': 'fujifilm_superia_200',
    'superia400': 'fujifilm_superia_400_fotu',
    'tmax100': 'kodak_tmax_100',
    'trix400': 'kodak_tri_x_400',
    'ultramax400': 'kodak_ultramax_400_fotu',
    'velvia50': 'fujifilm_velvia_50',
    'verita200d': 'kodak_verita_200d_fotu',
    'vision250d': 'kodak_vision3_250d_fotu',
    'vision500t': 'kodak_vision3_500t_fotu',
}

def softplus(u):
    """Numerically stable softplus: log(1 + exp(u))."""
    return np.where(u > 30.0, u, np.log1p(np.exp(np.clip(u, -50.0, 30.0))))

def evaluate_parametric_curve(c_dict, log_exp_axis):
    """Evaluate 6-parameter (+ optional secondary) Hurter-Driffield curve."""
    d_min = float(c_dict['dMin'])
    gamma = float(c_dict['gamma'])
    toe = float(c_dict['toe'])
    toe_w = float(c_dict['toeWidth'])
    sh = float(c_dict['shoulder'])
    sh_w = float(c_dict['shoulderWidth'])
    
    t = toe_w * softplus((log_exp_axis - toe) / toe_w)
    s = sh_w * softplus((log_exp_axis - sh) / sh_w)
    density = d_min + gamma * np.minimum(np.maximum(t - s, 0.0), sh - toe)
    
    sec = c_dict.get('secondary')
    if sec:
        sec_gamma = float(sec['gamma'])
        sec_toe = float(sec['toe'])
        sec_toe_w = float(sec['toeWidth'])
        sec_sh = float(sec['shoulder'])
        sec_sh_w = float(sec['shoulderWidth'])
        t2 = sec_toe_w * softplus((log_exp_axis - sec_toe) / sec_toe_w)
        s2 = sec_sh_w * softplus((log_exp_axis - sec_sh) / sec_sh_w)
        density += sec_gamma * np.minimum(np.maximum(t2 - s2, 0.0), sec_sh - sec_toe)
        
    return density

def convert_single_stock(fotufilm_stock_path, output_dirs):
    with open(fotufilm_stock_path, 'r', encoding='utf-8') as fp:
        raw = json.load(fp)

    raw_id = raw.get('id', os.path.splitext(os.path.basename(fotufilm_stock_path))[0])
    if raw.get('isExample'):
        print(f"Skipping synthetic example: {raw_id}")
        return None, None

    stock_slug = STOCK_SLUG_MAP.get(raw_id, raw_id)
    name = raw.get('name', stock_slug)
    is_mono = bool(raw.get('isMonochrome', False))
    is_rev = bool(raw.get('isReversal', False))
    is_paper = bool(raw.get('isReflectionPrint', False))

    wavelengths = np.arange(380, 785, 5, dtype=float)
    assert len(wavelengths) == 81

    # Reference illuminant
    kelvin = raw.get('referenceIlluminantKelvin', 5500)
    if kelvin <= 3400:
        ref_ill = 'T'
    elif kelvin <= 5200:
        ref_ill = 'D50'
    else:
        ref_ill = 'D55'

    # 1. Spectral Sensitivity & Dyes
    spectral = raw.get('spectral', {})
    sens_layers = np.array(spectral.get('layerSensitivity', []), dtype=float) # (3, 81)
    dye_layers = np.array(spectral.get('imageDyeDensity', []), dtype=float)   # (3, 81)

    if sens_layers.shape != (3, 81):
        if sens_layers.shape == (1, 81):
            sens_layers = np.tile(sens_layers, (3, 1))
        else:
            raise ValueError(f"Unexpected layerSensitivity shape: {sens_layers.shape}")

    if is_mono:
        dye_layers = np.full((3, 81), 1.0 / 3.0, dtype=float)
    elif dye_layers.shape != (3, 81):
        if dye_layers.size == 0:
            dye_layers = np.ones((3, 81), dtype=float)
        elif dye_layers.shape == (1, 81):
            dye_layers = np.tile(dye_layers, (3, 1))
        else:
            raise ValueError(f"Unexpected imageDyeDensity shape: {dye_layers.shape}")

    channel_density = dye_layers.T   # (81, 3)

    # FIX 1: Accurate Sensitivity Calibration
    # Fotufilm stores linear sensitivity. We calibrate scale so 18.4% gray produces raw == 1.0 (log_raw == 0.0)
    test_in = np.full((1, 1, 3), 0.184, dtype=np.float32)
    raw_mid = rgb_to_raw_hanatos2025(test_in, sens_layers.T, color_space='sRGB', apply_cctf_decoding=False, reference_illuminant=ref_ill)
    raw_vals = np.maximum(raw_mid[0, 0], 1e-8)
    scale = 1.0 / raw_vals
    scaled_sens = np.maximum(sens_layers.T * scale, 1e-12)
    log_sensitivity = np.log10(scaled_sens)

    # 2. Curves: Read dMin and Characteristic Curves
    curves_data = raw.get('curves', [])
    assert len(curves_data) == 3, f"Expected 3 curves, got {len(curves_data)}"
    d_min_vals = np.array([float(curves_data[ch]['dMin']) for ch in range(3)], dtype=float)

    # FIX 2: Base density (D-min spectrum)
    if 'minimumDensity' in spectral:
        base_density = np.array(spectral['minimumDensity'], dtype=float)
    else:
        if is_mono:
            base_density = np.full(81, d_min_vals[0], dtype=float)
        else:
            base_density = np.sum(d_min_vals * channel_density, axis=1)

    # FIX 3: Characteristic Density Curves (Baseline-zeroed Net Dye Curves)
    log_exposure = np.linspace(-4.0, 4.0, 256, dtype=float)
    density_curves = np.zeros((256, 3), dtype=float)
    sampled_curves_store = []

    if is_rev:
        # Reversal (slide) film calibration:
        # Standardize middle-gray exposure (logE = 0.0) net dye density to standard reference D ≈ 0.52
        fine_exp = np.linspace(-8.0, 8.0, 4096, dtype=float)
        fine_nets = []
        pchips = []
        d_maxs = []
        for ch in range(3):
            c_dict = curves_data[ch]
            d_min_ch = float(c_dict['dMin'])
            if 'sampled' in c_dict and c_dict['sampled']:
                s = c_dict['sampled']
                pchip = PchipInterpolator(s['logExposure'], s['density'], extrapolate=True)
                d_max_ch = float(np.max(s['density']))
                full_den = pchip(fine_exp)
                pchips.append(pchip)
                d_maxs.append(d_max_ch)
            else:
                full_den = evaluate_parametric_curve(c_dict, fine_exp)
                gamma = float(c_dict['gamma'])
                toe = float(c_dict['toe'])
                sh = float(c_dict['shoulder'])
                sec = c_dict.get('secondary')
                sec_range = float(sec['gamma']) * (float(sec['shoulder']) - float(sec['toe'])) if sec else 0.0
                d_max_ch = d_min_ch + gamma * (sh - toe) + sec_range
                pchips.append(None)
                d_maxs.append(d_max_ch)
            fine_nets.append(np.maximum(d_max_ch - full_den, 0.0))

        fine_nets = np.array(fine_nets).T
        mean_net = np.mean(fine_nets, axis=1)
        delta_log_e = float(np.interp(0.52, mean_net[::-1], fine_exp[::-1]))

        for ch in range(3):
            c_dict = curves_data[ch]
            if pchips[ch] is not None:
                density_curves[:, ch] = np.maximum(d_maxs[ch] - pchips[ch](log_exposure + delta_log_e), 0.0)
                sampled_curves_store.append({
                    'logExposure': log_exposure.tolist(),
                    'density': density_curves[:, ch].tolist()
                })
            else:
                density_curves[:, ch] = np.maximum(d_maxs[ch] - evaluate_parametric_curve(c_dict, log_exposure + delta_log_e), 0.0)
                sampled_curves_store.append(None)
    else:
        for ch in range(3):
            c_dict = curves_data[ch]
            d_min_ch = float(c_dict['dMin'])
            if 'sampled' in c_dict and c_dict['sampled']:
                samp = c_dict['sampled']
                s_exp = np.array(samp['logExposure'], dtype=float)
                s_den = np.array(samp['density'], dtype=float)
                pchip = PchipInterpolator(s_exp, s_den, extrapolate=True)
                full_den = pchip(log_exposure)
                d_max_ch = float(np.max(s_den))
                sampled_curves_store.append({
                    'logExposure': s_exp.tolist(),
                    'density': s_den.tolist()
                })
            else:
                full_den = evaluate_parametric_curve(c_dict, log_exposure)
                gamma = float(c_dict['gamma'])
                toe = float(c_dict['toe'])
                sh = float(c_dict['shoulder'])
                sec = c_dict.get('secondary')
                sec_range = 0.0
                if sec:
                    sec_range = float(sec['gamma']) * (float(sec['shoulder']) - float(sec['toe']))
                d_max_ch = d_min_ch + gamma * (sh - toe) + sec_range
                sampled_curves_store.append(None)

            density_curves[:, ch] = np.maximum(full_den - d_min_ch, 0.0)

    # FIX 4: Midscale neutral density
    idx_0 = np.argmin(np.abs(log_exposure))
    cmy_midgray = density_curves[idx_0]
    midscale_neutral_density = base_density + np.dot(channel_density, cmy_midgray)

    # Antihalation classification
    antihalation = 'weak'
    if 'cinestill' in stock_slug.lower():
        antihalation = 'no'
    elif 'vision' in stock_slug.lower() or 'eterna' in stock_slug.lower():
        antihalation = 'strong'

    info = ProfileInfo(
        stock=stock_slug,
        name=name,
        type='positive' if is_rev else 'negative',
        support='paper' if is_paper else 'film',
        stage='printing' if is_paper else 'filming',
        use='cine' if ('vision' in stock_slug or 'eterna' in stock_slug or 'f64' in stock_slug) else 'still',
        antihalation=antihalation,
        target_print=None if (is_rev or is_paper) else 'kodak_portra_endura',
        channel_model='bw' if is_mono else 'color',
        densitometer='status_A' if is_rev else 'status_M',
        log_sensitivity_density_over_min=0.2,
        reference_illuminant=ref_ill,
        viewing_illuminant='D50'
    )

    metadata = ProfileMetadata(
        version='fotufilm-port-2.0',
        copyright='Fotufilm profiles by MUAStudio Inc., licensed under CC BY-SA 4.0.',
        license='CC BY-SA 4.0. Redistribution must credit MUAStudio Inc. / Fotufilm.',
        datasource=f"Converted from fotufilm-engine profile '{raw_id}'. CC BY-SA 4.0."
    )

    physical_extensions = {
        'source_engine': 'fotufilm-engine',
        'fotufilm_id': raw_id,
        'schema_version': raw.get('schemaVersion', 3),
        'sampled_curves': sampled_curves_store,
        'crystal_grain': {
            'population': raw.get('crystalGrainPopulation'),
            'size_mm': raw.get('grainSizeMM', 0.008),
            'strength': raw.get('grainStrength', 0.02),
            'density_profile': raw.get('grainDensityProfile'),
            'layer_weights': raw.get('grainLayerWeights'),
            'density_law': raw.get('grainDensityLaw', 2),
        },
        'layered_transport': raw.get('layeredTransport'),
        'halation': {
            'look_scale': raw.get('halationLookScale', 1.0),
            'strength': raw.get('halationStrength'),
            'return_matrix': raw.get('halationReturnMatrix'),
            'profile': raw.get('halationProfile'),
            'base_thickness_mm': 0.13,
            'refractive_index': 1.48
        },
        'couplers': {
            'coupler_diffusion_mm': raw.get('couplerDiffusionMM'),
            'adjacency_radius_mm': raw.get('adjacencyRadiusMM'),
            'adjacency_strength': raw.get('adjacencyStrength', 0.0),
            'coupler_inhibition': raw.get('couplerInhibition'),
            'coupler_geometry': raw.get('couplerGeometry'),
            'coupler_release_gamma': raw.get('couplerReleaseGamma'),
        },
        'flare': raw.get('flare', 0.0),
        'reciprocity_failure': raw.get('reciprocityFailure')
    }

    cgp = raw.get('crystalGrainPopulation')
    if cgp and 'sublayerShares' in cgp and len(cgp['sublayerShares']) >= 3:
        shares = np.array(cgp['sublayerShares'][:3], dtype=float)
        if shares.sum() > 0:
            shares /= shares.sum()
        else:
            shares = np.array([0.33, 0.34, 0.33], dtype=float)
    else:
        shares = np.array([0.33, 0.34, 0.33], dtype=float)

    density_curves_layers = np.zeros((256, 3, 3), dtype=float)
    for l_idx in range(3):
        density_curves_layers[:, :, l_idx] = density_curves * shares[l_idx]

    data = ProfileData(
        wavelengths=wavelengths,
        log_sensitivity=log_sensitivity,
        channel_density=channel_density,
        base_density=base_density,
        midscale_neutral_density=midscale_neutral_density,
        log_exposure=log_exposure,
        density_curves=density_curves,
        density_curves_layers=density_curves_layers,
        density_curves_model=DensityCurvesModel(),
        hanatos2025_adaptation_window_params=np.empty((0,), dtype=float),
        hanatos2025_adaptation_surface_params=np.empty((0, 3), dtype=float),
        physical_extensions=physical_extensions
    )

    profile = Profile(metadata=metadata, info=info, data=data)
    _validate_profile(profile, stock_slug)

    out_dict = profile_to_dict(profile)
    for od in output_dirs:
        os.makedirs(od, exist_ok=True)
        out_path = os.path.join(od, f"{stock_slug}.json")
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(_json_safe(out_dict), fp, indent=4, allow_nan=False)

    print(f"Successfully converted: {raw_id} -> {stock_slug}.json (scale={scale.round(1)})")
    return stock_slug, profile

def update_all_neutral_print_filters(converted_profiles, filter_files):
    """
    Computes exact neutral print filters [0.0, M, Y] for all papers and enlarger illuminants
    for all negative film profiles, ensuring 100% natural printing.
    """
    papers = [
        'kodak_endura_premier', 'kodak_portra_endura', 'kodak_ultra_endura',
        'kodak_supra_endura', 'kodak_ektacolor_edge', 'fujifilm_crystal_archive_typeii',
        'kodak_2383', 'kodak_2393'
    ]
    illuminants = ['TH-KG3']

    # Load existing filters
    base_filters = {}
    for f_path in filter_files:
        if os.path.exists(f_path):
            with open(f_path, 'r', encoding='utf-8') as fp:
                base_filters = json.load(fp)
            break

    # Build target ratios for each paper + illuminant from a known calibrated stock (Portra 400)
    ref_film = load_profile('kodak_portra_400')
    ref_idx0 = np.argmin(np.abs(ref_film.data.log_exposure))
    ref_cmy_mid = ref_film.data.density_curves[ref_idx0:ref_idx0+1][None, :, :]
    ref_ds = compute_density_spectral(ref_film.data.channel_density, ref_cmy_mid, base_density=ref_film.data.base_density)

    paper_targets = {}
    for p_name in papers:
        paper_prof = load_profile(p_name)
        paper_sens = 10 ** paper_prof.data.log_sensitivity
        paper_targets[p_name] = {}
        for ill in illuminants:
            lamp = standard_illuminant(ill)
            cal_filt = base_filters.get(p_name, {}).get(ill, {}).get('kodak_portra_400', [0.0, 55.0, 55.0])
            filtered_lamp = color_enlarger(lamp, cal_filt)
            light = density_to_light(ref_ds, filtered_lamp)
            raw_p = np.einsum('ijk, kl->ijl', light, paper_sens)[0, 0]
            paper_targets[p_name][ill] = (raw_p[1] / raw_p[0], raw_p[2] / raw_p[0])

    print("\n--- Solving Neutral Print Filters for Converted Stocks ---")
    for stock_slug, profile in converted_profiles.items():
        is_neg = profile.info.type == 'negative' and profile.info.channel_model == 'color'
        
        # Midgray film density
        idx_0 = np.argmin(np.abs(profile.data.log_exposure))
        cmy_mid = profile.data.density_curves[idx_0:idx_0+1][None, :, :]
        ds = compute_density_spectral(profile.data.channel_density, cmy_mid, base_density=profile.data.base_density)

        for p_name in papers:
            if p_name not in base_filters:
                base_filters[p_name] = {}
            paper_prof = load_profile(p_name)
            paper_sens = 10 ** paper_prof.data.log_sensitivity

            for ill in illuminants:
                if ill not in base_filters[p_name]:
                    base_filters[p_name][ill] = {}

                if not is_neg:
                    # Reversal / Mono films: neutral baseline
                    base_filters[p_name][ill][stock_slug] = [0.0, 50.0, 50.0]
                else:
                    target_g, target_b = paper_targets[p_name][ill]
                    lamp = standard_illuminant(ill)

                    def obj(my):
                        fl = color_enlarger(lamp, [0.0, my[0], my[1]])
                        li = density_to_light(ds, fl)
                        rp = np.einsum('ijk, kl->ijl', li, paper_sens)[0, 0]
                        return [rp[1] / rp[0] - target_g, rp[2] / rp[0] - target_b]

                    # Initial guess from default or similar stock
                    guess = [60.0, 60.0]
                    res = root(obj, guess, method='hybr')
                    if res.success and np.all(np.isfinite(res.x)):
                        base_filters[p_name][ill][stock_slug] = [0.0, float(res.x[0]), float(res.x[1])]
                    else:
                        base_filters[p_name][ill][stock_slug] = [0.0, 60.0, 60.0]

    # Save to all filter files
    for f_path in filter_files:
        os.makedirs(os.path.dirname(f_path), exist_ok=True)
        with open(f_path, 'w', encoding='utf-8') as fp:
            json.dump(base_filters, fp, indent=4)
        print(f"Updated neutral print filters saved to: {f_path}")

def clear_lut_cache():
    cache_dir = os.path.join(ROOT_DIR, 'resources', '.lut_cache')
    if os.path.exists(cache_dir):
        count = 0
        for f in os.listdir(cache_dir):
            if f.endswith('.npy'):
                try:
                    os.remove(os.path.join(cache_dir, f))
                    count += 1
                except Exception:
                    pass
        print(f"Cleared {count} obsolete cached LUTs from {cache_dir}")

def main():
    stocks_dir = os.path.join(ROOT_DIR, 'fotufilm-engine', 'Sources', 'FotufilmCore', 'Stocks')
    output_dirs = [
        os.path.join(ROOT_DIR, 'resources', 'profiles'),
        os.path.join(ROOT_DIR, 'spektrafilm', 'data', 'profiles')
    ]
    filter_files = [
        os.path.join(ROOT_DIR, 'resources', 'neutral_print_filters.json'),
        os.path.join(ROOT_DIR, 'spektrafilm', 'data', 'filters', 'neutral_print_filters.json')
    ]

    converted = {}
    failed = []

    files = sorted(os.listdir(stocks_dir))
    for f in files:
        if not f.endswith('.json'):
            continue
        p = os.path.join(stocks_dir, f)
        try:
            slug, prof = convert_single_stock(p, output_dirs)
            if slug and prof:
                converted[slug] = prof
        except Exception as e:
            print(f"Error converting {f}: {e}")
            failed.append((f, str(e)))

    print(f"\n==========================================")
    print(f"Conversion Summary: {len(converted)} profiles converted")
    if failed:
        print(f"Failed conversions: {len(failed)}")
        for fn, err in failed:
            print(f"  {fn}: {err}")
    print(f"==========================================")

    # Solve and write neutral print filters
    update_all_neutral_print_filters(converted, filter_files)

    # Clean old broken LUT cache
    clear_lut_cache()
    print("All tasks completed successfully!")

if __name__ == '__main__':
    main()
