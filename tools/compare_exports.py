"""
Comprehensive quantitative comparison script between Fotufilm and Spektra Darkroom exports.
"""
import os
import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_filter

def load_img(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def main():
    path_fotu = os.path.abspath('对比/IMG_8430-fotufilm.com-ultramax400.jpg')
    path_spek = os.path.abspath('对比/IMG_8430-spektra-ultramax400.jpg')

    print(f"Loading Fotufilm: {path_fotu}")
    rgb_fotu = load_img(path_fotu)
    print(f"Loading Spektra:  {path_spek}")
    rgb_spek = load_img(path_spek)

    h, w, c = rgb_fotu.shape
    print(f"Image dimensions: {w}x{h}, {c} channels\n")

    # 1. Global RGB statistics
    print("=" * 60)
    print("1. GLOBAL RGB CHANNEL STATISTICS")
    print("=" * 60)
    channels = ['Red', 'Green', 'Blue']
    for i, ch in enumerate(channels):
        f_ch = rgb_fotu[:, :, i].astype(np.float32)
        s_ch = rgb_spek[:, :, i].astype(np.float32)
        diff = s_ch - f_ch
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        corr = np.corrcoef(f_ch.ravel(), s_ch.ravel())[0, 1]
        print(f"[{ch}]")
        print(f"  Fotufilm : Mean={f_ch.mean():.2f}, Std={f_ch.std():.2f}, Min={f_ch.min():.0f}, Max={f_ch.max():.0f}")
        print(f"  Spektra  : Mean={s_ch.mean():.2f}, Std={s_ch.std():.2f}, Min={s_ch.min():.0f}, Max={s_ch.max():.0f}")
        print(f"  Delta(S-F): Mean={diff.mean():+.2f}, MAE={mae:.2f}, RMSE={rmse:.2f}, Pearson r={corr:.5f}")

    # Global Image Quality Metrics
    # Sample down or step for faster SSIM if needed, but 4032x3024 can be done directly or on L channel
    print("\n" + "=" * 60)
    print("2. PERCEPTUAL SIMILARITY METRICS")
    print("=" * 60)
    img_psnr = psnr(rgb_fotu, rgb_spek)
    print(f"PSNR (Global RGB): {img_psnr:.2f} dB")

    # Convert to Lab (normalized 0..1)
    lab_fotu = rgb2lab(rgb_fotu.astype(np.float32) / 255.0)
    lab_spek = rgb2lab(rgb_spek.astype(np.float32) / 255.0)

    # Subsample Lab for CIEDE2000 calculation if needed (or calculate on full resolution with step 2)
    step = 2
    de2000 = deltaE_ciede2000(lab_fotu[::step, ::step], lab_spek[::step, ::step])
    print(f"CIEDE2000 Color Difference (sampled step={step}):")
    print(f"  Mean DeltaE : {np.mean(de2000):.3f}")
    print(f"  Median      : {np.median(de2000):.3f}")
    print(f"  Std         : {np.std(de2000):.3f}")
    print(f"  Min / Max   : {np.min(de2000):.3f} / {np.max(de2000):.3f}")
    print(f"  p75 / p90   : {np.percentile(de2000, 75):.3f} / {np.percentile(de2000, 90):.3f}")
    print(f"  p95 / p99   : {np.percentile(de2000, 95):.3f} / {np.percentile(de2000, 99):.3f}")

    # Delta L, a, b
    dL = lab_spek[:, :, 0] - lab_fotu[:, :, 0]
    da = lab_spek[:, :, 1] - lab_fotu[:, :, 1]
    db = lab_spek[:, :, 2] - lab_fotu[:, :, 2]
    print(f"\nLab Component Deltas (Spektra - Fotufilm):")
    print(f"  Delta L* (Lightness) : Mean={np.mean(dL):+.2f}, Std={np.std(dL):.2f}")
    print(f"  Delta a* (Green-Red) : Mean={np.mean(da):+.2f}, Std={np.std(da):.2f}")
    print(f"  Delta b* (Blue-Yellow): Mean={np.mean(db):+.2f}, Std={np.std(db):.2f}")

    # SSIM on L channel
    ssim_val = ssim(lab_fotu[::2, ::2, 0], lab_spek[::2, ::2, 0], data_range=100.0)
    print(f"SSIM (Luminance L*): {ssim_val:.4f}")

    # 3. Luminance Zone Analysis
    print("\n" + "=" * 60)
    print("3. TONAL ZONE BREAKDOWN (Based on Fotufilm L*)")
    print("=" * 60)
    zones = [
        ("Deep Shadows", 0, 15),
        ("Shadows", 15, 30),
        ("Midtones Low", 30, 50),
        ("Midtones High", 50, 70),
        ("Highlights Low", 70, 85),
        ("Highlights High", 85, 95),
        ("Extreme Highlights", 95, 101)
    ]
    f_L = lab_fotu[:, :, 0]
    s_L = lab_spek[:, :, 0]

    for name, low, high in zones:
        mask = (f_L >= low) & (f_L < high)
        count = np.sum(mask)
        pct = count / mask.size * 100.0
        if count > 0:
            zone_dL = s_L[mask] - f_L[mask]
            zone_da = da[mask]
            zone_db = db[mask]
            print(f"[{name}] (L* in [{low}, {high}), {pct:.1f}% pixels):")
            print(f"  dL* Mean={np.mean(zone_dL):+.2f}, da* Mean={np.mean(zone_da):+.2f}, db* Mean={np.mean(zone_db):+.2f}")
            print(f"  Fotufilm L* Mean={np.mean(f_L[mask]):.2f}, Spektra L* Mean={np.mean(s_L[mask]):.2f}")

    # 4. Highlight Roll-off and Clipping Analysis
    print("\n" + "=" * 60)
    print("4. HIGHLIGHT ROLL-OFF & CLIPPING CHECK")
    print("=" * 60)
    # Check max values and clipping percent
    f_clip_255 = np.mean(rgb_fotu == 255) * 100.0
    s_clip_255 = np.mean(rgb_spek == 255) * 100.0
    print(f"Pixels clipped at 255 in any channel:")
    print(f"  Fotufilm: {f_clip_255:.3f}%")
    print(f"  Spektra : {s_clip_255:.3f}%")
    print(f"Max RGB Fotufilm: R={rgb_fotu[:,:,0].max()}, G={rgb_fotu[:,:,1].max()}, B={rgb_fotu[:,:,2].max()}")
    print(f"Max RGB Spektra : R={rgb_spek[:,:,0].max()}, G={rgb_spek[:,:,1].max()}, B={rgb_spek[:,:,2].max()}")
    print(f"Max L* Fotufilm : {f_L.max():.2f}")
    print(f"Max L* Spektra  : {s_L.max():.2f}")

    # 5. Grain Structure & Noise Analysis
    print("\n" + "=" * 60)
    print("5. GRAIN STRUCTURE & NOISE RMS ANALYSIS")
    print("=" * 60)
    # High-pass filter to extract grain noise
    # We blur with Gaussian sigma=2.0 and subtract from original to isolate grain
    print("Extracting high-frequency grain textures (sigma=2.0 high-pass filter)...")
    sigma = 2.0
    f_hp = f_L - gaussian_filter(f_L, sigma=sigma)
    s_hp = s_L - gaussian_filter(s_L, sigma=sigma)

    # Compute grain RMS across tonal zones
    grain_zones = [
        ("Shadows", 10, 30),
        ("Midtones", 30, 60),
        ("Bright Midtones", 60, 75),
        ("Highlights", 75, 88),
        ("Extreme Highlights", 88, 100)
    ]
    print(f"{'Zone':<20} | {'Fotufilm Grain RMS':<20} | {'Spektra Grain RMS':<20} | {'Ratio S/F':<10}")
    print("-" * 75)
    f_rms_list = []
    s_rms_list = []
    for name, low, high in grain_zones:
        mask = (f_L >= low) & (f_L < high)
        if np.sum(mask) > 1000:
            f_rms = np.std(f_hp[mask])
            s_rms = np.std(s_hp[mask])
            ratio = s_rms / (f_rms + 1e-6)
            f_rms_list.append(f_rms)
            s_rms_list.append(s_rms)
            print(f"{name:<20} | {f_rms:<20.3f} | {s_rms:<20.3f} | {ratio:<10.3f}")

    # Relative roll-off: ratio of Highlight grain to Midtone grain
    f_rolloff = f_rms_list[-1] / (f_rms_list[1] + 1e-6)
    s_rolloff = s_rms_list[-1] / (s_rms_list[1] + 1e-6)
    print(f"\nHighlight Roll-off Suppression (Ext.Highlight RMS / Midtone RMS):")
    print(f"  Fotufilm Highlight Suppression Ratio : {f_rolloff:.3f} ({(1-f_rolloff)*100:.1f}% reduction)")
    print(f"  Spektra Highlight Suppression Ratio  : {s_rolloff:.3f} ({(1-s_rolloff)*100:.1f}% reduction)")

    # Color Grain Analysis (Chroma noise std)
    f_a_hp = lab_fotu[:,:,1] - gaussian_filter(lab_fotu[:,:,1], sigma=sigma)
    f_b_hp = lab_fotu[:,:,2] - gaussian_filter(lab_fotu[:,:,2], sigma=sigma)
    s_a_hp = lab_spek[:,:,1] - gaussian_filter(lab_spek[:,:,1], sigma=sigma)
    s_b_hp = lab_spek[:,:,2] - gaussian_filter(lab_spek[:,:,2], sigma=sigma)

    f_chroma_rms = np.sqrt(np.mean(f_a_hp**2 + f_b_hp**2))
    s_chroma_rms = np.sqrt(np.mean(s_a_hp**2 + s_b_hp**2))
    print(f"\nChroma Grain RMS (Color Noise):")
    print(f"  Fotufilm Chroma RMS : {f_chroma_rms:.3f}")
    print(f"  Spektra Chroma RMS  : {s_chroma_rms:.3f}")
    print(f"  Chroma Noise Ratio (S/F): {s_chroma_rms / f_chroma_rms:.3f}")

if __name__ == '__main__':
    main()
