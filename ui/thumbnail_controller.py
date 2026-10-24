"""SpektraDarkroom Thumbnail Controller & Lifecycle Manager
Unified architecture for:
1. Bottom filmstrip thumbnails (dual-layer: immutable raw base + dynamic graded WYSIWYG)
2. Left sidebar stock cards (clean brand color pills vs dynamic candidate previews)
3. Batch recipe thumbnail grading and instantaneous filmstrip refreshing
"""

import os
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread, Qt
from PySide6.QtGui import QImage, QPixmap
from core import session_cache_manager
import logging
logger = logging.getLogger("SpektraDarkroom.ThumbnailController")


def apply_post_lut_adjustments_f32(img_f32: np.ndarray, params: dict) -> np.ndarray:
    """Vectorized CPU post-LUT adjustments strictly aligned with GLSL shader & export pipeline:
    - DIR Coupler inhibition response (Color separation & samelayer contrast)
    - Enlarger Color Timing (CMY filters, print exposure, preflash)
    - Paper Chemistry Morph & Developer Exhaustion
    """
    if img_f32 is None or not params:
        return img_f32

    out = img_f32.copy()

    # 1. DIR Coupler inhibition response (Color separation & samelayer contrast)
    dir_intl = float(params.get("dir_interlayer", 1.0))
    dir_same = float(params.get("dir_samelayer", 1.0))
    if abs(dir_intl - 1.0) > 0.01 or abs(dir_same - 1.0) > 0.01:
        luma_f = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
        chroma = out - luma_f[..., None]
        out = luma_f[..., None] + chroma * dir_intl
        out = np.power(np.clip(out, 0.0001, 1.0), 1.0 / max(0.2, dir_same))

    # 2. Enlarger Color Timing (CMY filters, print exposure, preflash) - bypassed only for positive slide film
    film = str(params.get("film_stock", "")).lower()
    is_pos = ("positive" in film) or ("reversal" in film) or ("e100" in film) or ("velvia" in film) or ("provia" in film)
    if not is_pos:
        c_val = float(params.get("enlarger_cyan", 0.0))
        m_val = float(params.get("enlarger_magenta", 0.0))
        y_val = float(params.get("enlarger_yellow", 0.0))
        print_exp = float(params.get("print_exposure", 1.0))
        preflash = float(params.get("pre_flash", 0.0))
        if abs(c_val) > 0.01 or abs(m_val) > 0.01 or abs(y_val) > 0.01 or abs(print_exp - 1.0) > 0.01 or abs(preflash) > 0.001:
            cmy_factor = 1.0 - np.array([c_val, m_val, y_val], dtype=np.float32) * 0.015
            out = np.power(np.clip(out, 0.0001, 1.0), cmy_factor) * print_exp + preflash

        # 3. Paper Chemistry Morph & Developer Exhaustion
        morph_gamma = float(params.get("morph_gamma", 1.0))
        dev_exh = float(params.get("developer_exhaustion", 0.0))
        if abs(morph_gamma - 1.0) > 0.01:
            out = np.power(np.clip(out, 0.0001, 1.0), morph_gamma)
        if dev_exh > 0.005:
            out = out * (1.0 - dev_exh * 0.6) + (out * 0.88 + 0.03) * (dev_exh * 0.6)

    return np.clip(out, 0.0, 1.0)


class StockThumbnailWorker(QThread):
    """Background worker generating candidate 3D LUT previews for unselected film & paper cards."""
    film_ready = Signal(str, QPixmap)
    paper_ready = Signal(str, QPixmap)

    def __init__(self, engine, linear_thumb, film_stocks, paper_stocks, current_film, current_paper,
                 do_films=True, do_papers=True, ev=0.0, highlights=0.0, shadows=0.0, base_ev=0.0,
                 params_dict=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        # Store reference without deepcopying large arrays synchronously on GUI thread
        self.linear_thumb = linear_thumb
        # Prioritize active stocks first for snappy responsive rendering
        f_list = list(film_stocks)
        if current_film in f_list:
            f_list.remove(current_film)
            f_list.insert(0, current_film)
        self.film_stocks = f_list

        p_list = list(paper_stocks)
        if current_paper in p_list:
            p_list.remove(current_paper)
            p_list.insert(0, current_paper)
        self.paper_stocks = p_list

        self.current_film = current_film
        self.current_paper = current_paper
        self.do_films = do_films
        self.do_papers = do_papers
        self.ev = float(ev)
        self.highlights = float(highlights)
        self.shadows = float(shadows)
        self.base_ev = float(base_ev)
        self.params_dict = dict(params_dict) if params_dict else {}
        self._stopped = False

    def stop(self):
        self._stopped = True
        self.linear_thumb = None

    def run(self):
        if self.linear_thumb is None or self._stopped:
            return

        from core.app_core import prophoto_to_srgb_filmic, srgb_to_linear, SRGB_TO_PROPHOTO

        raw_input = self.linear_thumb
        # If passed an 8-bit integer image, accurately convert sRGB uint8 to linear ProPhoto float32
        if raw_input.dtype == np.uint8:
            srgb_norm = raw_input.astype(np.float32) / 255.0
            lin_srgb = srgb_to_linear(srgb_norm)
            small_float = np.maximum(0.0, lin_srgb @ SRGB_TO_PROPHOTO.T)
        elif raw_input.dtype == np.float32:
            small_float = raw_input
        else:
            small_float = raw_input.astype(np.float32)

        # Scale down while strictly preserving aspect ratio so 3D LUT grading is ultra-fast
        h, w = small_float.shape[:2]
        max_edge = 120
        if max(h, w) > max_edge:
            scale = max_edge / float(max(h, w))
            target_w = max(1, int(round(w * scale)))
            target_h = max(1, int(round(h * scale)))
            small_linear = cv2.resize(small_float, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            small_linear = small_float.copy()

        # Immediately release full-resolution reference so worker only holds tiny 120px array in RAM
        self.linear_thumb = None
        del raw_input, small_float

        # 1. Apply exposure EV & Scene-Linear Highlights / Shadows (Strictly aligned with GLSL u_exposure)
        total_ev = self.ev + self.base_ev
        if abs(total_ev) > 0.001:
            small_linear = small_linear * (2.0 ** total_ev)

        # 2. Apply White Balance (CCT & Tint) aligned with GLSL shader & CPU export
        base_temp = float(self.params_dict.get("base_temp", 5500.0))
        base_tint = float(self.params_dict.get("base_tint", 1.0))
        temp_val = float(self.params_dict.get("color_temp", base_temp))
        tint_val = float(self.params_dict.get("tint", base_tint))
        if abs(temp_val - base_temp) > 0.1 or abs(tint_val - base_tint) > 0.001:
            temp_factor = (temp_val - base_temp) / 3000.0
            tint_factor = tint_val / max(0.001, base_tint)
            small_linear[..., 0] *= (1.0 + temp_factor * 0.22)
            small_linear[..., 2] *= (1.0 - temp_factor * 0.22)
            small_linear[..., 1] *= tint_factor

        # 3. Apply Highlights & Shadows tone curve
        hl_val = float(self.params_dict.get("highlights", self.highlights))
        sh_val = float(self.params_dict.get("shadows", self.shadows))
        if abs(hl_val) > 0.001 or abs(sh_val) > 0.001:
            try:
                from core.app_core import apply_highlights_shadows_np
                small_linear = apply_highlights_shadows_np(small_linear, hl_val, sh_val)
            except Exception as e:
                logger.error(f"Error applying highlights/shadows: {e}")

        # 3.5. Apply Optical Diffusion filter if enabled
        diff_family = str(self.params_dict.get("diffusion_family", "none")).lower()
        diff_strength = float(self.params_dict.get("diffusion_strength", 0.0))
        diff_warmth = float(self.params_dict.get("diffusion_warmth", 0.0))
        if diff_family != "none" and diff_strength > 0.001:
            try:
                from core.app_core import apply_diffusion_np
                small_linear = apply_diffusion_np(small_linear, diff_family, diff_strength, diff_warmth)
            except Exception as e:
                logger.error(f"Error applying optical diffusion: {e}")

        # Reference stocks to pair with when current stock is 'none' (bypass mode)
        ref_paper = self.current_paper
        ref_film = self.current_film if self.current_film != "none" else "kodak_portra_400"

        if self._stopped:
            return

        # 4. Process Film stocks (paired with current paper or reference paper)
        if self.do_films:
            for sid in self.film_stocks:
                if self._stopped:
                    return
                try:
                    if sid == "none":
                        srgb_f32 = prophoto_to_srgb_filmic(small_linear)
                        graded_f32 = apply_post_lut_adjustments_f32(srgb_f32, self.params_dict)
                        srgb_u8 = (np.clip(graded_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
                        gh, gw, gc = srgb_u8.shape
                        qimg = QImage(srgb_u8.data, gw, gh, gw * gc, QImage.Format.Format_RGB888).copy()
                        pix = QPixmap.fromImage(qimg)
                        self.film_ready.emit(sid, pix)
                    else:
                        p_dict = dict(self.params_dict)
                        if ref_paper == "none" and "print_mode" not in p_dict:
                            p_dict["print_mode"] = "scan"
                        lut = self.engine.get_3d_lut(sid, ref_paper, lut_size=17, params_dict=p_dict)
                        graded_f32 = self.engine.apply_lut_to_linear(small_linear, lut, as_float=True)
                        graded_f32 = apply_post_lut_adjustments_f32(graded_f32, self.params_dict)
                        graded_u8 = (np.clip(graded_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
                        gh, gw, gc = graded_u8.shape
                        qimg = QImage(graded_u8.data, gw, gh, gw * gc, QImage.Format.Format_RGB888).copy()
                        pix = QPixmap.fromImage(qimg)
                        self.film_ready.emit(sid, pix)
                except Exception:
                    pass
                # Cooperatively yield GIL to GUI thread so mouse, scroll, and clicks stay at 60 FPS
                if not self._stopped:
                    QThread.msleep(4)

        if self._stopped:
            return

        # 5. Process Paper stocks (paired with current film or reference film)
        if self.do_papers:
            for sid in self.paper_stocks:
                if self._stopped:
                    return
                try:
                    if sid == "none":
                        # Paper "none" represents Digital Film Scan (print_mode="scan") using negative scan inversion LUT
                        lut = self.engine.get_3d_lut(ref_film, "none", lut_size=17, params_dict=self.params_dict, print_mode="scan")
                        graded_f32 = self.engine.apply_lut_to_linear(small_linear, lut, as_float=True)
                        graded_f32 = apply_post_lut_adjustments_f32(graded_f32, self.params_dict)
                        graded_u8 = (np.clip(graded_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
                        gh, gw, gc = graded_u8.shape
                        qimg = QImage(graded_u8.data, gw, gh, gw * gc, QImage.Format.Format_RGB888).copy()
                        pix = QPixmap.fromImage(qimg)
                        self.paper_ready.emit(sid, pix)
                    else:
                        lut = self.engine.get_3d_lut(ref_film, sid, lut_size=17, params_dict=self.params_dict)
                        graded_f32 = self.engine.apply_lut_to_linear(small_linear, lut, as_float=True)
                        graded_f32 = apply_post_lut_adjustments_f32(graded_f32, self.params_dict)
                        graded_u8 = (np.clip(graded_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
                        gh, gw, gc = graded_u8.shape
                        qimg = QImage(graded_u8.data, gw, gh, gw * gc, QImage.Format.Format_RGB888).copy()
                        pix = QPixmap.fromImage(qimg)
                        self.paper_ready.emit(sid, pix)
                except Exception:
                    pass
                # Cooperatively yield GIL to GUI thread so mouse, scroll, and clicks stay at 60 FPS
                if not self._stopped:
                    QThread.msleep(4)


class ThumbnailController(QObject):
    """Centralized orchestrator for all thumbnail lifecycles:
    - Guarantees immutable preservation of raw negative base thumbnails.
    - Manages dynamic graded thumbnails without data loss.
    - Controls sidebar card states (clean brand color pills vs live candidate previews).
    - Accelerates batch recipe application with instant filmstrip synchronization.
    """

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._card_worker = None

    # =========================================================================
    # 1. Bottom Filmstrip Thumbnail Management (Dual-Layer: Raw Base + Graded)
    # =========================================================================

    @staticmethod
    def get_raw_thumbnail(photo: dict) -> np.ndarray | None:
        """Retrieves the pristine, immutable raw base thumbnail (0..255 uint8)."""
        if not photo or not isinstance(photo, dict):
            return None

        _is_valid = session_cache_manager.is_valid_thumbnail

        raw = photo.get("raw_thumbnail_rgb")
        if _is_valid(raw):
            return raw

        raw = photo.get("raw_thumbnail")
        if _is_valid(raw):
            photo["raw_thumbnail_rgb"] = raw
            return raw

        # Attempt recovery from session cache
        p_path = photo.get("path")
        if p_path:
            try:
                cached = session_cache_manager.get_photo_cache(p_path)
                if cached and _is_valid(cached.get("raw_thumbnail_rgb")):
                    raw = cached["raw_thumbnail_rgb"]
                    photo["raw_thumbnail_rgb"] = raw
                    photo["raw_thumbnail"] = raw
                    return raw
            except Exception:
                pass

        # Attempt recovery from float_img preview
        float_img = photo.get("float_img")
        if float_img is not None and isinstance(float_img, np.ndarray) and float_img.size > 0:
            try:
                import cv2
                from core.app_core import prophoto_to_srgb_filmic
                th_h = 72
                th_w = max(48, int(float_img.shape[1] * (th_h / float(float_img.shape[0]))))
                th_img = cv2.resize(float_img, (th_w, th_h), interpolation=cv2.INTER_AREA)
                raw = (prophoto_to_srgb_filmic(th_img) * 255.0).astype(np.uint8)
                if _is_valid(raw):
                    photo["raw_thumbnail_rgb"] = raw
                    photo["raw_thumbnail"] = raw
                    return raw
            except Exception:
                pass

        if _is_valid(photo.get("thumbnail_rgb")) and not photo.get("is_edited") and not photo.get("params"):
            raw = photo["thumbnail_rgb"]
            photo["raw_thumbnail_rgb"] = np.copy(raw)
            photo["raw_thumbnail"] = np.copy(raw)
            return raw

        # Emergency fallback: standalone fast thumbnail extraction
        if p_path:
            try:
                raw = session_cache_manager.extract_fast_thumbnail(p_path)
                if _is_valid(raw):
                    photo["raw_thumbnail_rgb"] = raw
                    photo["raw_thumbnail"] = raw
                    return raw
            except Exception:
                pass
        # Fallback: If no real thumbnail could be healed (e.g. synthetic test photo),
        # return existing array if available rather than returning None
        for cand in (photo.get("raw_thumbnail_rgb"), photo.get("raw_thumbnail"), photo.get("thumbnail_rgb")):
            if cand is not None and isinstance(cand, np.ndarray) and cand.size > 0:
                photo["raw_thumbnail_rgb"] = cand
                photo["raw_thumbnail"] = cand
                return cand

        return None

    @staticmethod
    def set_raw_thumbnail(photo: dict, raw_u8: np.ndarray):
        """Stores the immutable raw base thumbnail."""
        if photo is not None and isinstance(photo, dict) and raw_u8 is not None:
            c_arr = np.ascontiguousarray(raw_u8, dtype=np.uint8)
            photo["raw_thumbnail_rgb"] = c_arr
            photo["raw_thumbnail"] = c_arr
            if photo.get("thumbnail_rgb") is None:
                photo["thumbnail_rgb"] = c_arr

    @staticmethod
    def get_display_thumbnail(photo: dict) -> np.ndarray | None:
        """Returns the active display thumbnail: graded if edited, otherwise raw base."""
        if not photo:
            return None
        graded = photo.get("graded_thumbnail_rgb")
        if graded is not None and isinstance(graded, np.ndarray) and graded.size > 0:
            return graded
        return ThumbnailController.get_raw_thumbnail(photo)

    def revert_photo_to_raw(self, photo: dict, filmstrip_widget=None) -> bool:
        """Reverts a photo's displayed thumbnail completely back to the unedited raw base thumbnail.
        Fired when '删除暗房配置 (.sdc)' is invoked.
        """
        if not photo:
            return False
        photo["graded_thumbnail_rgb"] = None
        photo["is_edited"] = False
        photo["is_dirty"] = False
        photo["film_profile"] = "none"
        photo["paper_profile"] = "none"
        photo.pop("params", None)

        raw_thumb = self.get_raw_thumbnail(photo)
        if raw_thumb is None or raw_thumb.size == 0:
            return False

        photo["thumbnail_rgb"] = raw_thumb
        photo["raw_thumbnail_rgb"] = raw_thumb
        photo["raw_thumbnail"] = raw_thumb
        if filmstrip_widget and photo.get("id"):
            c_arr = np.ascontiguousarray(raw_thumb, dtype=np.uint8)
            h, w, c = c_arr.shape
            qimg = QImage(c_arr.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            filmstrip_widget.update_photo_thumbnail(photo["id"], pix)

        try:
            session_cache_manager.save_photo_cache(photo.get("path"), photo, save_preview=False)
        except Exception:
            pass
        return True

    def update_graded_thumbnail(self, photo: dict, rendered_rgb: np.ndarray, filmstrip_widget=None) -> bool:
        """Updates the photo's graded thumbnail from a valid offscreen canvas render."""
        if not photo or rendered_rgb is None or rendered_rgb.size == 0:
            return False

        # Guard: Reject pitch-black or uninitialized buffer
        if rendered_rgb.mean() < 1.0 and rendered_rgb.max() < 10:
            logger.debug("[ThumbnailController] Refusing to update thumbnail with black buffer")
            return False

        photo["graded_thumbnail_rgb"] = rendered_rgb
        photo["thumbnail_rgb"] = rendered_rgb

        if filmstrip_widget and photo.get("id"):
            c_arr = np.ascontiguousarray(rendered_rgb, dtype=np.uint8)
            h, w, c = c_arr.shape
            qimg = QImage(c_arr.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            filmstrip_widget.update_photo_thumbnail(photo["id"], pix)

        try:
            session_cache_manager.save_photo_cache(photo.get("path"), photo, save_preview=False)
        except Exception:
            pass
        return True

    def grade_thumbnail_fast(self, raw_thumb: np.ndarray, film="none", paper="none", params=None) -> np.ndarray:
        """CPU thumbnail grading applying exposure, white balance, highlights/shadows, 3D LUT, and post-LUT adjustments."""
        if raw_thumb is None or not hasattr(self, "engine"):
            return raw_thumb
        params = params or {}
        try:
            from core.app_core import srgb_to_linear, SRGB_TO_PROPHOTO, prophoto_to_srgb_filmic
            if raw_thumb.dtype == np.uint8:
                srgb_norm = raw_thumb.astype(np.float32) / 255.0
                lin_srgb = srgb_to_linear(srgb_norm)
                small_linear = np.maximum(0.0, lin_srgb @ SRGB_TO_PROPHOTO.T)
            else:
                small_linear = raw_thumb.astype(np.float32)

            ev = float(params.get("exposure_ev", 0.0)) + float(params.get("base_ev", 0.0))
            if abs(ev) > 0.001:
                small_linear = small_linear * (2.0 ** ev)

            # White balance (CCT & Tint)
            base_temp = float(params.get("base_temp", 5500.0))
            base_tint = float(params.get("base_tint", 1.0))
            temp_val = float(params.get("color_temp", base_temp))
            tint_val = float(params.get("tint", base_tint))
            if abs(temp_val - base_temp) > 0.1 or abs(tint_val - base_tint) > 0.001:
                temp_factor = (temp_val - base_temp) / 3000.0
                tint_factor = tint_val / max(0.001, base_tint)
                small_linear[..., 0] *= (1.0 + temp_factor * 0.22)
                small_linear[..., 2] *= (1.0 - temp_factor * 0.22)
                small_linear[..., 1] *= tint_factor

            hl = float(params.get("highlights", 0.0))
            sh = float(params.get("shadows", 0.0))
            if abs(hl) > 0.001 or abs(sh) > 0.001:
                try:
                    from core.app_core import apply_highlights_shadows_np
                    small_linear = apply_highlights_shadows_np(small_linear, hl, sh)
                except Exception:
                    pass

            # Optical diffusion filter
            diff_family = str(params.get("diffusion_family", "none")).lower()
            diff_strength = float(params.get("diffusion_strength", 0.0))
            diff_warmth = float(params.get("diffusion_warmth", 0.0))
            if diff_family != "none" and diff_strength > 0.001:
                try:
                    from core.app_core import apply_diffusion_np
                    small_linear = apply_diffusion_np(small_linear, diff_family, diff_strength, diff_warmth)
                except Exception:
                    pass

            if film == "none" and paper == "none":
                out_f32 = prophoto_to_srgb_filmic(small_linear)
            else:
                p_dict = dict(params)
                if paper == "none" and "print_mode" not in p_dict:
                    p_dict["print_mode"] = "scan"
                lut = self.engine.get_3d_lut(film, paper, lut_size=17, params_dict=p_dict)
                out_f32 = self.engine.apply_lut_to_linear(small_linear, lut, as_float=True)

            out_f32 = apply_post_lut_adjustments_f32(out_f32, params)
            return (np.clip(out_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        except Exception:
            return raw_thumb

    def batch_apply_recipe_thumbnails(self, photos_list: list, film: str, paper: str, params: dict, filmstrip_widget=None):
        """Immediately applies a darkroom recipe to a batch of target photos, refreshing filmstrip thumbnails."""
        for photo in photos_list:
            if not photo:
                continue
            raw_thumb = self.get_raw_thumbnail(photo)
            if raw_thumb is None or raw_thumb.size == 0:
                continue

            graded = self.grade_thumbnail_fast(raw_thumb, film=film, paper=paper, params=params)
            photo["graded_thumbnail_rgb"] = graded
            photo["thumbnail_rgb"] = graded

            if filmstrip_widget and photo.get("id"):
                c_arr = np.ascontiguousarray(graded, dtype=np.uint8)
                h, w, c = c_arr.shape
                qimg = QImage(c_arr.data, w, h, w * c, QImage.Format.Format_RGB888).copy()
                filmstrip_widget.update_photo_thumbnail(photo["id"], QPixmap.fromImage(qimg))
                filmstrip_widget.set_photo_dirty(photo["id"], True)

            try:
                session_cache_manager.save_photo_cache(photo.get("path"), photo, save_preview=False)
            except Exception:
                pass

    # =========================================================================
    # 2. Left Sidebar Stock Cards (Films & Papers)
    # =========================================================================

    def reset_all_cards_to_color_pills(self, film_cards: dict, paper_cards: dict):
        """Completely purges all photographic previews from sidebar cards and restores
        clean official brand color swatches (Color Pills). Fired on '删除暗房配置' or empty state.
        """
        self.cancel_card_preview_worker()

        # 1. Reset all film cards: clear preview image, set default to 'none' (数码原生)
        if film_cards:
            for sid, card in film_cards.items():
                card.set_thumbnail(None)
                card.set_active(sid == "none")

        # 2. Reset all paper cards: clear preview image, set default to 'none' (数码扫描)
        if paper_cards:
            for sid, card in paper_cards.items():
                card.set_thumbnail(None)
                card.set_active(sid == "none")

    def cancel_card_preview_worker(self):
        """Safely stops any active background candidate preview generator without blocking GUI thread."""
        if self._card_worker:
            worker = self._card_worker
            self._card_worker = None
            try:
                worker.film_ready.disconnect()
            except Exception:
                pass
            try:
                worker.paper_ready.disconnect()
            except Exception:
                pass
            if hasattr(worker, "isRunning") and worker.isRunning():
                worker.stop()
                worker.wait(100)
                if worker.isRunning():
                    if not hasattr(self, "_retained_workers"):
                        self._retained_workers = []
                    self._retained_workers.append(worker)
                    try:
                        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))
                    except Exception:
                        pass

    def _cleanup_worker(self, worker):
        if hasattr(self, "_retained_workers") and worker in self._retained_workers:
            try:
                self._retained_workers.remove(worker)
            except Exception:
                pass

    def update_active_card_wysiwyg(self, film_cards: dict, paper_cards: dict,
                                   active_film: str, active_paper: str, pixmap: QPixmap, force: bool = False):
        """Updates active cards only if uninitialized or explicitly forced; preserves crisp CPU candidate thumbnails."""
        if pixmap and not pixmap.isNull():
            if film_cards and active_film in film_cards:
                card = film_cards[active_film]
                if force or card.thumbnail_pixmap is None:
                    card.set_thumbnail(pixmap)
            if paper_cards and active_paper in paper_cards:
                card = paper_cards[active_paper]
                if force or card.thumbnail_pixmap is None:
                    card.set_thumbnail(pixmap)

    def trigger_candidate_card_previews(self, linear_thumb: np.ndarray, film_cards: dict, paper_cards: dict,
                                        current_film: str, current_paper: str, params: dict, base_ev: float = 0.0,
                                        on_film_ready=None, on_paper_ready=None):
        """Starts background worker generating candidate previews for unselected film & paper cards."""
        self.cancel_card_preview_worker()
        if linear_thumb is None or linear_thumb.size == 0:
            return

        ev = float(params.get("exposure_ev", 0.0))
        hl = float(params.get("highlights", 0.0))
        sh = float(params.get("shadows", 0.0))
        eff_base_ev = float(base_ev) if abs(base_ev) > 0.0001 else float(params.get("base_ev", 0.0))

        self._card_worker = StockThumbnailWorker(
            self.engine, linear_thumb,
            list(film_cards.keys()) if film_cards else [],
            list(paper_cards.keys()) if paper_cards else [],
            current_film, current_paper,
            do_films=bool(film_cards),
            do_papers=bool(paper_cards),
            ev=ev, highlights=hl, shadows=sh, base_ev=eff_base_ev,
            params_dict=params,
            parent=self
        )
        if on_film_ready:
            self._card_worker.film_ready.connect(on_film_ready)
        if on_paper_ready:
            self._card_worker.paper_ready.connect(on_paper_ready)
        self._card_worker.start()
