from __future__ import annotations

import numpy as np
from opt_einsum import contract

from spektrafilm.model.diffusion import apply_diffusion_filter_um
from spektrafilm.model.develop import compute_density_spectral, develop_print_morph
from spektrafilm.model.illuminants import standard_illuminant
from spektrafilm.utils.conversions import density_to_light


class PrintingStage:
    def __init__(
        self,
        film,
        film_render_params,
        print_profile,
        print_render_params,
        enlarger_params,
        settings_params,
        lut_service,
        enlarger_service,
        resize_service,
        color_reference_service,
    ):
        self._film = film
        self._film_render = film_render_params
        self._print = print_profile
        self._print_render = print_render_params
        self._enlarger = enlarger_params
        self._settings = settings_params
        self._lut_service = lut_service
        self._enlarger_service = enlarger_service
        self._resize_service = resize_service
        self._color_reference_service = color_reference_service

    # public methods

    def expose(self, cmy_film_density: np.ndarray) -> np.ndarray:
        
        cmy_film_black = np.zeros((1,1,3)) - np.array(self._film_render.grain.density_min)
        cmy_film_white = np.nanmax(self._film.data.density_curves, axis=0)[None, None, :]
        self._color_reference_service.log_raw_print_black = self._film_cmy_to_print_log_raw(cmy_film_black)
        self._color_reference_service.log_raw_print_white = self._film_cmy_to_print_log_raw(cmy_film_white)
        
        log_raw_print = self._lut_service.spectral_compute_enlarger(
            cmy_film_density,
            spectral_calculation=self._film_cmy_to_print_log_raw,
            data_min=-np.array(self._film_render.grain.density_min),
            data_max=np.nanmax(self._film.data.density_curves, axis=0),
            use_lut=self._settings.use_enlarger_lut,
        )    
        raw = 10**log_raw_print
        raw *= self._enlarger.print_exposure
        raw *= self._color_reference_service.black_white_printing_exposure_correction()
        raw = apply_diffusion_filter_um(
            raw,
            self._enlarger.diffusion_filter,
            pixel_size_um=self._resize_service.pixel_size_um,
        )
        return np.log10(np.fmax(raw, 0.0) + 1e-10)

    def develop(self, log_raw: np.ndarray) -> np.ndarray:
                
        return develop_print_morph(
            log_raw,
            self._print.data.log_exposure,
            self._print.data.density_curves_model,
            density_curves_morph=self._print_render.density_curves_morph,
            profile_type=self._print.info.type,
        )

    # private methods

    def _film_cmy_to_print_log_raw(self, cmy_film_density: np.ndarray) -> np.ndarray:
        """Print exposure, structured like the reference engine's.

        The spectral integration runs under the RAW enlarger lamp — the reference prints a
        reflection paper through unfiltered 3200 K tungsten (PrintIllumination.swift) and
        neutralizes per paper record at its own calibrated midpoint.  We had been integrating
        through the solved dichroic filtration, which shapes the lamp and starves whichever
        paper record the filtration cuts (Y58 starves blue): that record's paper curve falls
        into its toe and its grain dies on the tones it drives (measured on a sky tone:
        per-record grain transfer B/R 0.55 filtered vs 4.48 raw vs the reference's 1.18 — the
        lamp is the dominant control).

        So: energies integrate under the raw lamp; the per-record operating point comes from an
        anchor solved at mid-grey — ``anchor[c]`` is where the *filtered* calibration put
        mid-grey on the paper's exposure axis, i.e. the filtration's neutrality work survives as
        a per-record log-exposure offset (the equivalent of the reference's paperMidpoint) while
        the tone-to-tone grain crosstalk flows through the raw spectral response.  Mid-grey
        lands exactly where the previous chain put it, so neutrality and the display
        calibration built on it are preserved.
        """
        sensitivity = 10 ** self._print.data.log_sensitivity
        sensitivity = np.nan_to_num(sensitivity)
        enlarger_light_source = standard_illuminant(self._enlarger.illuminant)

        anchor, log_mid_raw = self._print_anchor(sensitivity, enlarger_light_source)

        density_spectral = compute_density_spectral(
            self._film.data.channel_density,
            cmy_film_density,
            base_density=self._film.data.base_density,
        )
        light = density_to_light(density_spectral, enlarger_light_source)
        raw = contract("ijk, kl->ijl", light, sensitivity)
        # Relative to mid-grey under the same raw lamp, then onto the anchored operating point.
        relative = (np.log10(np.fmax(raw, 1e-12))
                    - log_mid_raw)
        log_raw = anchor + relative
        if self._enlarger.preflash_exposure > 0:
            # Uniform flash in the relative-energy domain, as the reference applies it.
            flash = self._raw_preflash_energy(sensitivity, enlarger_light_source)
            log_raw = np.log10(10.0 ** relative + flash / np.fmax(
                10.0 ** log_mid_raw, 1e-12)) + anchor
        return log_raw

    def _print_anchor(self, sensitivity, enlarger_light_source):
        """(anchor, log mid-grey raw energy), both per record; computed once per pipeline run.

        The anchor is the previous (filtered-integration) chain's own mid-grey landing — the
        dichroic filtration keeps its neutrality role exactly there and nowhere else.
        """
        cached = getattr(self, "_print_anchor_cache", None)
        if cached is not None:
            return cached
        filtered_lamp = self._enlarger_service.enlarger_filtered_illuminant(
            enlarger_light_source)
        mid_spectral = self._enlarger_service.density_spectral_midgray

        # Where the filtered calibration put mid-grey, per record (the old chain's own value).
        light_f = density_to_light(mid_spectral, filtered_lamp)
        e_f_mid = contract("ijk, kl->ijl", light_f, sensitivity)
        factor_f = self._compute_exposure_factor_midgray(sensitivity, filtered_lamp)
        preflash_f = self._compute_raw_preflash(enlarger_light_source, sensitivity)
        anchor = np.log10(np.fmax(e_f_mid * factor_f + preflash_f, 1e-10))

        # Raw-lamp mid-grey energies, per record, for the relative axis.
        light_r = density_to_light(mid_spectral, enlarger_light_source)
        e_r_mid = contract("ijk, kl->ijl", light_r, sensitivity)
        log_mid_raw = np.log10(np.fmax(e_r_mid, 1e-12))

        self._print_anchor_cache = (anchor, log_mid_raw)
        return self._print_anchor_cache

    def _raw_preflash_energy(self, sensitivity, enlarger_light_source):
        """Preflash energy integrated under the raw lamp (absolute units)."""
        if self._enlarger.preflash_exposure <= 0:
            return np.zeros((3,))
        preflash_illuminant = self._enlarger_service.preflash_filtered_illuminant(
            enlarger_light_source)
        density_base = np.asarray(self._film.data.base_density)[None, None, :]
        light_preflash = density_to_light(density_base, preflash_illuminant)
        raw_preflash = contract("ijk, kl->ijl", light_preflash, sensitivity)
        return raw_preflash * self._enlarger.preflash_exposure

    def _compute_raw_preflash(self, light_source, sensitivity):
        if self._enlarger.preflash_exposure > 0:
            preflash_illuminant = self._enlarger_service.preflash_filtered_illuminant(light_source)
            density_base = np.asarray(self._film.data.base_density)[None, None, :]
            light_preflash = density_to_light(density_base, preflash_illuminant)
            raw_preflash = contract("ijk, kl->ijl", light_preflash, sensitivity)
            return raw_preflash * self._enlarger.preflash_exposure
        return np.zeros((3,))

    def _compute_exposure_factor_midgray(self, sensitivity, print_illuminant):
        factor_midgray = _exposure_factor(sensitivity, print_illuminant, self._enlarger_service.density_spectral_midgray)
        if self._enlarger_service.density_spectral_midgray_comp is not None:
            factor_midgray_comp = _exposure_factor(sensitivity, print_illuminant,
                                                    self._enlarger_service.density_spectral_midgray_comp)
        else:
            factor_midgray_comp = 1.0
        if self._enlarger.print_exposure_compensation and not self._enlarger.normalize_print_exposure:
            return factor_midgray_comp / factor_midgray
        elif self._enlarger.normalize_print_exposure and self._enlarger.print_exposure_compensation:
            return factor_midgray_comp
        elif self._enlarger.normalize_print_exposure and not self._enlarger.print_exposure_compensation:
            return factor_midgray
        else:
            return 1.0

def _exposure_factor(sensitivity, print_illuminant, density_spectral_midgray):
    light_midgray = density_to_light(density_spectral_midgray, print_illuminant)
    raw_midgray = contract("ijk, kl->ijl", light_midgray, sensitivity)
    raw_midgray = np.fmax(raw_midgray, 1e-10)
    # use the geometric mean to normalize the exposure
    raw_midgray_geomean = np.exp(np.mean(np.log(raw_midgray), axis=2, keepdims=True))
    return 1 / raw_midgray_geomean