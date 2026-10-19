import numpy as np
import scipy
import scipy.ndimage
from spektrafilm.model.density_curves import interp_density_cmy_layers
from spektrafilm.runtime.params_schema import GrainParams
from spektrafilm.utils.fast_stats import fast_binomial, fast_poisson, fast_lognormal_from_mean_std
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter

################################################################################
# Grain (very simple model)
################################################################################

def layer_particle_model(density,
                         density_max=2.2,
                         n_particles_per_pixel=10,
                         grain_uniformity=0.98,
                         seed=None,
                         blur_particle=0.0,
                         method='poisson_binomial',
                         use_fast_stats=False,
                         ):
    if seed is not None:
        np.random.seed(seed) # scipy uses np.random
    
    probability_of_development = density/density_max
    probability_of_development = np.clip(probability_of_development, 1e-6, 1-1e-6) # for safe calc
    od_particle = density_max/n_particles_per_pixel
    
    grain = np.zeros_like(density)
    if method=='gamma_beta':
        gamma_rvs = scipy.stats.gamma.rvs
        beta_rvs = scipy.stats.beta.rvs
        seeds = gamma_rvs(n_particles_per_pixel/(1-grain_uniformity+1e-6), size=density.shape) * (1-grain_uniformity+1e-6)
        grain = beta_rvs(probability_of_development*n_particles_per_pixel,
                        (1-probability_of_development)*n_particles_per_pixel)*seeds*od_particle
    elif method=='poisson_binomial':
        if use_fast_stats:
            binom_rvs = fast_binomial
            poisson_rvs = fast_poisson
        else:
            binom_rvs = scipy.stats.binom.rvs
            poisson_rvs = scipy.stats.poisson.rvs
        saturation = 1 - probability_of_development*grain_uniformity*(1-1e-6)
        seeds = poisson_rvs(n_particles_per_pixel/saturation)
        grain = binom_rvs(seeds, probability_of_development)
        grain = np.double(grain)*od_particle*saturation
    
    if blur_particle>0:
        # grain = scipy.ndimage.gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
        grain = fast_gaussian_filter(grain, blur_particle*np.sqrt(od_particle))
    return grain

def add_micro_structure(density_cmy_out, micro_structure, pixel_size_um):
    grain_micro_structure_blur_pixel = micro_structure[0]/pixel_size_um
    grain_micro_structure_sigma = micro_structure[1]*0.001/pixel_size_um  # grain microstructure[1] is in nm
    if grain_micro_structure_sigma > 0.05:
        clumping = fast_lognormal_from_mean_std(np.ones_like(density_cmy_out),
                                                np.ones_like(density_cmy_out)*grain_micro_structure_sigma)
        if grain_micro_structure_blur_pixel>0.4:
            # clumping = scipy.ndimage.gaussian_filter(clumping, (grain_micro_structure_blur_pixel,
            #                                                     grain_micro_structure_blur_pixel, 0))
            clumping = fast_gaussian_filter(clumping, grain_micro_structure_blur_pixel)
        density_cmy_out *= clumping
    return density_cmy_out

# Jarvis 4-Gaussian mixture decomposition of exp(-r/k) dye cloud point spread
# (Jarvis, J. Photogr. Sci. 40:105, 1992, Eqs. 2 and 4; Graves & Saunders 1985)
JARVIS_DYE_CLOUD_GAUSSIANS = (
    (0.240, 0.518),
    (0.384, 0.983),
    (0.274, 1.848),
    (0.102, 3.731),
)


def apply_jarvis_dye_cloud_filter(density_field, decay_length_px):
    """Filter density field by the 4-Gaussian mixture decomposition of Jarvis's dye cloud exp(-r/k)."""
    if decay_length_px <= 0.2:
        return density_field
    out = np.zeros_like(density_field)
    for weight, sigma_factor in JARVIS_DYE_CLOUD_GAUSSIANS:
        sigma_px = max(decay_length_px * sigma_factor, 0.1)
        out += weight * fast_gaussian_filter(density_field, sigma_px)
    return out


def mix_chroma_grain(fluctuations_cmy, colour=0.5):
    """Variance-preserving mix between independent channel grain and shared luminance grain.

    Matches fotufilm Look.mix(colour):
      own = clamp(colour, 0.0, 1.0)
      shared = sqrt(max(0.0, 3.0 - 2.0 * own^2)) - own
      out = own * fluctuations + shared * mean_fluc
    When colour=1.0: pure independent dye clouds (chroma grain / 彩噪).
    When colour=0.0: pure shared luminance grain (单色明度颗粒), variance mathematically identical.
    """
    if fluctuations_cmy.ndim < 3 or fluctuations_cmy.shape[-1] != 3:
        return fluctuations_cmy
    own = float(np.clip(colour, 0.0, 1.0))
    if abs(own - 1.0) < 1e-4:
        return fluctuations_cmy
    shared = float(np.sqrt(max(0.0, 3.0 - 2.0 * own * own)) - own)
    mean_fluc = np.mean(fluctuations_cmy, axis=-1, keepdims=True)
    return own * fluctuations_cmy + shared * mean_fluc


def apply_grain_to_density(density_cmy,
                           pixel_size_um=10,
                           particle_area_um2=0.2,
                           particle_scale=[1,0.8,3],
                           density_min=[0.03,0.06,0.04],
                           density_max_curves=[2.2,2.2,2.2],
                           grain_uniformity=[0.98,0.98,0.98],
                           grain_blur=1.0,
                           n_sub_layers=1,
                           fixed_seed=None,
                           colour_grain=0.5,
                           ):
    density_min = np.array(density_min)
    density_max = density_max_curves + density_min
    pixel_area_um2 = pixel_size_um**2
    particle_area_um2 = particle_area_um2*np.array(particle_scale)
    n_particles_per_pixel = pixel_area_um2/particle_area_um2
    sigma_blur_pixel = grain_blur
    
    if fixed_seed is not None:
        seed = None
    else:
        seed = [0, 1, 2]
    
    if n_sub_layers>1:
        n_particles_per_pixel /= n_sub_layers
    
    density_cmy_base = density_cmy.copy()
    density_cmy += density_min
    density_cmy_out = np.zeros_like(density_cmy)
    for ch in np.arange(3):
        for sl in np.arange(n_sub_layers):
            density_cmy_out[:,:,ch] += layer_particle_model(density_cmy[:,:,ch],
                                                            density_max=density_max[ch],
                                                            n_particles_per_pixel=n_particles_per_pixel[ch],
                                                            grain_uniformity=grain_uniformity[ch],
                                                            seed=seed[ch] + sl*10)
    density_cmy_out /= n_sub_layers
    density_cmy_out -= density_min
    
    # Calculate pure fluctuation and apply variance-preserving chroma mix
    fluctuation = density_cmy_out - density_cmy_base
    fluctuation = mix_chroma_grain(fluctuation, colour=colour_grain)
    density_cmy_out = density_cmy_base + fluctuation

    if sigma_blur_pixel>0.4:
        density_cmy_out = fast_gaussian_filter(density_cmy_out, sigma_blur_pixel)
        
    return density_cmy_out


# experimental / multi-layer crystal grain
def apply_grain_to_density_layers(density_cmy_layers, # x,y,sublayers,rgb
                                  density_max_layers, # 3x3 [sublayers,rgb]
                                  pixel_size_um=10,
                                  particle_area_um2=0.2,
                                  particle_scale=[1,0.8,3], # rgb
                                  particle_scale_layers=[3,1,0.3], # sublayers
                                  density_min=[0.03,0.06,0.04],
                                  grain_uniformity=[0.98,0.98,0.98],
                                  grain_blur=1.0,
                                  grain_blur_dye_clouds_um=1.0,
                                  grain_micro_structure=(0.1, 30),
                                  fixed_seed=None,
                                  use_fast_stats=False,
                                  colour_grain=0.5,
                                  dye_cloud_decay_um=1.45,
                                  ):
    density_max_total = np.sum(density_max_layers, axis=0) # [sublayers,rgb]
    density_max_fractions = density_max_layers/density_max_total[None,:]
    density_min_layers = density_max_fractions*np.array(density_min)[None,:]
    density_max_layers = density_max_layers + density_min_layers
    
    pixel_area_um2 = pixel_size_um**2
    particle_area_um2_layers = (particle_area_um2 * 
                                    np.array(particle_scale)[None,:] * 
                                    np.array(particle_scale_layers)[:,None]) # layers, rgb
    n_particles_per_pixel = pixel_area_um2*density_max_fractions/particle_area_um2_layers

    if fixed_seed is not None:
        seed = None
    else:
        seed = [0, 1, 2]
    
    density_cmy_base = np.sum(density_cmy_layers.copy(), axis=2)
    density_cmy_layers += density_min_layers
    n_sl = min(density_cmy_layers.shape[2], density_max_layers.shape[0])
    density_cmy_out = np.zeros(density_cmy_layers.shape[0:2] + (3,))
    for ch in np.arange(3): # rgb channels
        for sl in np.arange(n_sl): # sublayers
            density_cmy_out[:,:,ch] += layer_particle_model(density_cmy_layers[:,:,sl,ch],
                                                            density_max=density_max_layers[sl,ch],
                                                            n_particles_per_pixel=n_particles_per_pixel[sl,ch],
                                                            grain_uniformity=grain_uniformity[ch],
                                                            seed=seed[ch] + sl*10,
                                                            blur_particle=grain_blur_dye_clouds_um,
                                                            use_fast_stats=use_fast_stats)
    
    # micro-structure
    density_cmy_out = add_micro_structure(density_cmy_out, grain_micro_structure, pixel_size_um)

    # final
    density_cmy_out -= density_min
    
    # Jarvis dye cloud filter
    decay_px = float(dye_cloud_decay_um or 1.45) / max(pixel_size_um, 0.1)
    if decay_px > 0.2:
        density_cmy_out = apply_jarvis_dye_cloud_filter(density_cmy_out, decay_px)

    # Variance-preserving chroma grain mix
    fluctuation = density_cmy_out - density_cmy_base
    fluctuation = mix_chroma_grain(fluctuation, colour=colour_grain)
    density_cmy_out = density_cmy_base + fluctuation

    if grain_blur>0:
        density_cmy_out = fast_gaussian_filter(density_cmy_out, grain_blur)
    return density_cmy_out


def apply_grain(
    density_cmy,
    pixel_size_um,
    grain: GrainParams,
    density_curves,
    density_curves_layers,
    profile_type,
    bypass_grain=False,
    use_fast_stats=False,
):
    if not grain.active or bypass_grain:
        return density_cmy

    colour_grain = getattr(grain, 'colour_grain', 0.5)
    crystal_grain = getattr(grain, 'crystal_grain', None) or {}

    has_valid_layers = (
        isinstance(density_curves_layers, np.ndarray) and
        density_curves_layers.ndim == 3 and
        density_curves_layers.shape[1] == 3 and
        density_curves_layers.shape[2] == 3
    )
    if not grain.sublayers_active or not has_valid_layers:
        density_max = np.nanmax(density_curves, axis=0)
        return apply_grain_to_density(
            density_cmy,
            pixel_size_um=pixel_size_um,
            particle_area_um2=grain.particle_area_um2,
            particle_scale=grain.particle_scale,
            density_min=grain.density_min,
            density_max_curves=density_max,
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            n_sub_layers=grain.n_sub_layers,
            colour_grain=colour_grain,
        )

    density_cmy_layers = interp_density_cmy_layers(
        density_cmy,
        density_curves,
        density_curves_layers,
        positive_film=profile_type == 'positive',
    )
    density_max_layers = np.nanmax(density_curves_layers, axis=0)
    
    particle_scale_layers = list(grain.particle_scale_layers)
    dye_cloud_decay_um = 1.45
    if crystal_grain:
        shares = crystal_grain.get('sublayer_shares')
        if shares and len(shares) >= 3:
            particle_scale_layers = [float(shares[0] * 6.0), float(shares[1] * 3.0), float(shares[2] * 1.5)]
        if 'dye_cloud_decay_mm' in crystal_grain:
            dye_cloud_decay_um = float(crystal_grain['dye_cloud_decay_mm']) * 1000.0

    return apply_grain_to_density_layers(
        density_cmy_layers,
        density_max_layers=density_max_layers,
        pixel_size_um=pixel_size_um,
        particle_area_um2=grain.particle_area_um2,
        particle_scale=grain.particle_scale,
        particle_scale_layers=particle_scale_layers,
        density_min=grain.density_min,
        grain_uniformity=grain.uniformity,
        grain_blur=grain.blur,
        grain_blur_dye_clouds_um=grain.blur_dye_clouds_um,
        grain_micro_structure=grain.micro_structure,
        use_fast_stats=use_fast_stats,
        colour_grain=colour_grain,
        dye_cloud_decay_um=dye_cloud_decay_um,
    )

# TODO: make grain parameter with RMS granularity

if __name__=='__main__':
    density = np.ones((128,128))*2
    g1 = layer_particle_model(density, density_max=2, n_particles_per_pixel=10, grain_uniformity=0.99, sigma_blur=0.)
    g2 = layer_particle_model(density, density_max=2, n_particles_per_pixel=10, grain_uniformity=0.96, sigma_blur=0.)
    print('g1 ------------------')
    print('Density Test')
    print('Mean', np.mean(g1))
    print('RMS', np.std(g1)*1000)
    print('Skewness', scipy.stats.skew(g1.flatten()))
    print('Kurtosis', scipy.stats.kurtosis(g1.flatten()))
    print('g2 ------------------')
    print('Mean', np.mean(g2))
    print('RMS', np.std(g2)*1000)
    print('Skewness', scipy.stats.skew(g2.flatten()))
    print('Kurtosis', scipy.stats.kurtosis(g2.flatten()))
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(g1, vmin=0, vmax=2.2)
    axs[0].set_title('Uniformity=0.99')
    axs[1].imshow(g2, vmin=0, vmax=2.2)
    axs[1].set_title('Uniformity=0.96')
    fig.suptitle('Fully saturated density with different uniformity')
    plt.show()
