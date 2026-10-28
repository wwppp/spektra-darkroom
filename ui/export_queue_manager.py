"""Export Queue Management System for SpektraDarkroom
Supports asynchronous background rendering with GPU FBO offscreen acceleration,
per-task cancellation, rapid cancel/clear safety, queue drawer UI, and live progress reporting.
"""

import os
import uuid
import logging
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread, Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QScrollArea, QWidget, QFrame
)
from PySide6.QtGui import QPixmap, QImage, QColor, QIcon, QFontMetrics

from ui.window_utils import apply_dark_titlebar
from ui.theme_manager import theme_manager
from ui.smooth_scroll import SmoothScrollArea

logger = logging.getLogger("SpektraDarkroom.ExportQueue")


def _save_rendered_file(rendered_arr, output_path, params_dict):
    """Encodes and saves the rendered image array to disk according to parameters,
    embedding complete camera EXIF metadata bundle and target colorspace ICC profile.
    """
    from core.app_core import save_encoded_image
    return save_encoded_image(rendered_arr, output_path, params_dict)


class ExportTask:
    def __init__(self, task_id_or_path, title_or_output=None, source_path=None, output_path=None,
                 params=None, thumbnail=None, **kwargs):
        if output_path is not None:
            # Called as ExportTask(tid, title, source_path, output_path, params, thumbnail=pix)
            self.task_id = str(task_id_or_path)
            from core.path_utils import normalize_path
            self.photo_path = normalize_path(source_path)
            self.output_path = normalize_path(output_path)
            self.filename = os.path.basename(self.output_path) if self.output_path else (str(title_or_output) if title_or_output else "image")
            self.params = dict(params or {})
            self.thumb_img = thumbnail
            self.bit_depth = self.params.get("bit_depth", 8)
            self.format_type = self.params.get("format", "jpeg")
        else:
            # Called as ExportTask(photo_path, output_path, params, filename=None, thumb_img=None, bit_depth=8, format_type="jpeg")
            from core.path_utils import normalize_path
            self.task_id = f"task_{uuid.uuid4().hex[:8]}"
            self.photo_path = normalize_path(task_id_or_path)
            self.output_path = normalize_path(title_or_output)
            self.params = dict(source_path or {})
            self.filename = kwargs.get("filename") or (os.path.basename(self.output_path) if self.output_path else (os.path.basename(self.photo_path) if self.photo_path else "image"))
            self.thumb_img = kwargs.get("thumb_img") if kwargs.get("thumb_img") is not None else thumbnail
            self.bit_depth = kwargs.get("bit_depth", self.params.get("bit_depth", 8))
            self.format_type = kwargs.get("format_type", self.params.get("format", "jpeg"))
        self.status = "queued"  # queued, running, finished, cancelled, failed
        self.progress = 0
        self.error_msg = ""


class _LoadImageWorker(QThread):
    finished_load = Signal(object, object, str)  # raw_target, lut_arr, error_msg

    def __init__(self, engine, photo_path, base_ev=0.0, params=None):
        super().__init__()
        self.engine = engine
        self.photo_path = photo_path
        self.base_ev = base_ev
        self.params = params or {}
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if self._is_cancelled:
            self.finished_load.emit(None, None, "任务已取消")
            return
        try:
            raw_target = None
            if hasattr(self.engine, "_load_full_resolution") and self.photo_path:
                try:
                    raw_target = self.engine._load_full_resolution(self.photo_path, base_ev=self.base_ev)
                except Exception as ex:
                    logger.warning(f"Full resolution load failed: {ex}")

            if raw_target is None and hasattr(self.engine, "load_image") and self.photo_path:
                res = self.engine.load_image(self.photo_path)
                if res.get("success"):
                    raw_target = self.engine.raw_preview

            if self._is_cancelled:
                self.finished_load.emit(None, None, "任务已取消")
                return

            lut_arr = None
            if raw_target is not None and hasattr(self.engine, "get_3d_lut") and self.params:
                film_stock = self.params.get("film_stock", "kodak_portra_400")
                paper_stock = self.params.get("paper_stock", "kodak_2383")
                print_mode = self.params.get("print_mode", "optical")
                try:
                    lut_arr = self.engine.get_3d_lut(film_stock, paper_stock, lut_size=33, print_mode=print_mode)
                except Exception as ex_lut:
                    logger.warning(f"Failed to fetch 3D LUT in background loader: {ex_lut}")

            if self._is_cancelled:
                self.finished_load.emit(None, None, "任务已取消")
            elif raw_target is not None:
                self.finished_load.emit(raw_target, lut_arr, "")
            else:
                self.finished_load.emit(None, None, "无法载入底片图像数据")
        except Exception as e:
            self.finished_load.emit(None, None, str(e))


class _SaveImageWorker(QThread):
    finished_save = Signal(bool, str)  # success, error_msg

    def __init__(self, rendered_arr, output_path, params_dict):
        super().__init__()
        self.rendered_arr = rendered_arr
        self.output_path = output_path
        self.params_dict = params_dict
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if self._is_cancelled:
            self.finished_save.emit(False, "任务已取消")
            return
        tmp_target = f"{self.output_path}.tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        try:
            _save_rendered_file(self.rendered_arr, tmp_target, self.params_dict)
            if self._is_cancelled:
                if os.path.exists(tmp_target):
                    try:
                        os.remove(tmp_target)
                    except Exception:
                        pass
                self.finished_save.emit(False, "任务已取消")
            else:
                os.replace(tmp_target, self.output_path)
                self.finished_save.emit(True, "")
        except Exception as e:
            if os.path.exists(tmp_target):
                try:
                    os.remove(tmp_target)
                except Exception:
                    pass
            self.finished_save.emit(False, str(e))


class _CpuExportWorker(QThread):
    progress = Signal(int)
    finished_export = Signal(bool, str)

    def __init__(self, engine, task):
        super().__init__()
        self.engine = engine
        self.task = task
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        def _prog(pct, msg=""):
            if not self._is_cancelled:
                self.progress.emit(max(10, min(95, int(pct))))

        def _cancel():
            return self._is_cancelled or self.task.status == "cancelled"

        tmp_target = f"{self.task.output_path}.tmp_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        try:
            res = self.engine.export_image(
                params_dict=self.task.params,
                output_path=tmp_target,
                format_type=self.task.format_type,
                quality=self.task.params.get("quality", 9),
                source_path=self.task.photo_path,
                progress_cb=_prog,
                cancel_cb=_cancel
            )
            if self._is_cancelled or self.task.status == "cancelled":
                if os.path.exists(tmp_target):
                    try:
                        os.remove(tmp_target)
                    except Exception:
                        pass
                self.finished_export.emit(False, "任务已取消")
            elif res.get("success"):
                os.replace(tmp_target, self.task.output_path)
                self.finished_export.emit(True, "")
            else:
                if os.path.exists(tmp_target):
                    try:
                        os.remove(tmp_target)
                    except Exception:
                        pass
                err = str(res.get("error") or "扫描显影失败")
                self.finished_export.emit(False, err)
        except Exception as e:
            if os.path.exists(tmp_target):
                try:
                    os.remove(tmp_target)
                except Exception:
                    pass
            self.finished_export.emit(False, str(e))


class ExportQueueManager(QObject):
    queue_changed = Signal()
    task_progress = Signal(str, int)    # task_id, progress (lightweight updater)
    status_changed = Signal(int, int)  # active_count, total_count
    status_updated = Signal(str, bool) # text, is_busy

    def __init__(self, engine, canvas=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.canvas = canvas
        self.tasks = []
        self._current_task = None
        self._current_worker = None
        self._is_active = True

    def set_canvas(self, canvas):
        self.canvas = canvas

    def enqueue(self, task):
        """Enqueue an ExportTask into the sequential processing queue."""
        if not isinstance(task, ExportTask):
            return
        self.tasks.append(task)
        self.queue_changed.emit()
        self._update_overall_status()
        QTimer.singleShot(0, self._process_next)

    def enqueue_batch(self, tasks):
        """Enqueue multiple ExportTasks into the queue with a single unified UI notification."""
        valid_tasks = [t for t in tasks if isinstance(t, ExportTask)]
        if not valid_tasks:
            return
        self.tasks.extend(valid_tasks)
        self.queue_changed.emit()
        self._update_overall_status()
        QTimer.singleShot(0, self._process_next)

    def add_task(self, photo_path, output_path, params, filename=None, thumb_img=None, bit_depth=8, format_type="jpeg"):
        task = ExportTask(photo_path, output_path, params, filename=filename, thumb_img=thumb_img,
                          bit_depth=bit_depth, format_type=format_type)
        self.enqueue(task)
        return task.task_id

    def _on_task_progress(self, task_id, pct):
        for t in self.tasks:
            if t.task_id == task_id:
                t.progress = pct
                break
        self.task_progress.emit(task_id, pct)

    def has_active_tasks(self):
        return self.get_active_count() > 0

    def get_active_count(self):
        return sum(1 for t in self.tasks if t.status in ("queued", "running"))

    def cancel_task(self, task_id):
        cancelled_current = False
        for task in self.tasks:
            if task.task_id == task_id:
                if task.status in ("queued", "running"):
                    task.status = "cancelled"
                    task.error_msg = "任务已取消"
                    if task is self._current_task:
                        cancelled_current = True
                        if self._current_worker and hasattr(self._current_worker, "cancel"):
                            try:
                                self._current_worker.cancel()
                            except Exception:
                                pass
                        # Deliberately keep _current_task / _current_worker bound until the
                        # worker actually stops. Clearing them here let _process_next start
                        # the next task while the cancelled one was still inside
                        # export_image(), i.e. two threads mutating one engine at once.
                    self.task_progress.emit(task.task_id, task.progress)
                    self.queue_changed.emit()
                    self._update_overall_status()
                break
        if cancelled_current:
            # The worker's own completion path advances the queue. This timer is only a
            # safety net for a worker that ignores cancel() entirely, so the queue cannot
            # stall forever behind a cancelled task.
            QTimer.singleShot(2000, self._process_next)

    def cancel_all(self):
        cancelled_any_current = False
        for task in self.tasks:
            if task.status in ("queued", "running"):
                task.status = "cancelled"
                task.error_msg = "任务已取消"
                self.task_progress.emit(task.task_id, task.progress)
                if task is self._current_task:
                    cancelled_any_current = True
        if self._current_worker and hasattr(self._current_worker, "cancel"):
            try:
                self._current_worker.cancel()
            except Exception:
                pass
        if cancelled_any_current:
            # Same rule as cancel_task: wait for the worker to come to rest instead of
            # dropping the reference and letting a second thread enter the engine.
            QTimer.singleShot(2000, self._process_next)
        self.queue_changed.emit()
        self._update_overall_status()

    def clear_history(self):
        if getattr(self, "_is_clearing", False):
            return
        self._is_clearing = True
        try:
            # Keep queued tasks and the actively running task so no worker reference breaks
            self.tasks = [t for t in self.tasks if t.status == "queued" or (t is self._current_task and t.status == "running")]
            self.queue_changed.emit()
            self._update_overall_status()
        finally:
            self._is_clearing = False

    def _process_next(self):
        if not self._is_active:
            return
        if self._current_task is not None:
            # If current task is already in inactive state, release lock immediately
            if self._current_task.status in ("cancelled", "finished", "failed"):
                self._current_task = None
                self._current_worker = None
            else:
                return

        next_task = None
        for t in self.tasks:
            if t.status == "queued":
                next_task = t
                break

        if not next_task:
            self._update_overall_status()
            return

        self._current_task = next_task
        next_task.status = "running"
        next_task.progress = 5
        self.queue_changed.emit()
        self.task_progress.emit(next_task.task_id, 5)
        self._update_overall_status()

        # Check Hardware Acceleration Mode
        hw_mode = "global"
        try:
            from core import config_manager
            hw_mode = config_manager.get_preferences().get("hardware_acceleration_mode", "global")
        except Exception:
            hw_mode = "global"

        # Hardware-accelerated GPU Pipeline:
        # Step 1 (Thread): Load 100% full-resolution RAW image in background
        # Step 2 (Main Thread): Render on GPU via FBO shader in ~20ms (zero freeze!)
        # Step 3 (Thread): Compress & write image to disk in background
        can_use_gpu = (
            hw_mode != "off"
            and self.canvas is not None
            and hasattr(self.canvas, "render_offscreen")
        )
        if can_use_gpu:
            self._start_gpu_export(next_task)
        else:
            self._start_cpu_export(next_task)

    def _start_gpu_export(self, task):
        task.params.setdefault("source_path", task.photo_path)
        task.params.setdefault("format", task.format_type or "jpeg")
        task.params.setdefault("bit_depth", task.bit_depth or 8)
        base_ev = float(task.params.get("base_ev", 0.0))
        worker = _LoadImageWorker(self.engine, task.photo_path, base_ev=base_ev, params=task.params)
        self._current_worker = worker

        def _on_load_done(raw_target, lut_arr, err_msg):
            if self._current_worker is worker:
                self._current_worker = None
            if task.status == "cancelled":
                self._finish_task(task, False, "任务已取消")
                return

            if err_msg or raw_target is None:
                # If RAW load failed or returned None, fallback to CPU pipeline
                self._start_cpu_export(task)
                return

            task.progress = 35
            self.task_progress.emit(task.task_id, 35)

            # Step 2: GPU FBO Offscreen Render (scheduled via QTimer to yield to Qt event loop)
            def _do_render():
                if task.status == "cancelled":
                    self._finish_task(task, False, "任务已取消")
                    return

                try:
                    from PySide6.QtCore import QCoreApplication
                    QCoreApplication.processEvents()

                    src_h, src_w = raw_target.shape[:2]
                    crop_cfg = task.params.get("crop", {})
                    if isinstance(crop_cfg, dict) and crop_cfg.get("enabled", False):
                        cw_factor = max(0.05, min(1.0, float(crop_cfg.get("w", 1.0))))
                        ch_factor = max(0.05, min(1.0, float(crop_cfg.get("h", 1.0))))
                    else:
                        cw_factor, ch_factor = 1.0, 1.0

                    # The dialog resolves its size preset against the real source geometry and
                    # hands over exact (width, height): use it verbatim, so the grain spec is
                    # derived at the size actually written and portrait frames stay portrait.
                    out_size = task.params.get("out_size") or None
                    tgt_w = tgt_h = 0
                    if isinstance(out_size, (list, tuple)) and len(out_size) == 2:
                        try:
                            tgt_w, tgt_h = int(out_size[0]), int(out_size[1])
                        except (TypeError, ValueError):
                            tgt_w = tgt_h = 0
                    if tgt_w <= 0 or tgt_h <= 0:
                        out_long_edge = int(task.params.get("out_long_edge", 0) or 0)
                        crop_long = max(src_w * cw_factor, src_h * ch_factor)
                        if out_long_edge > 0 and out_long_edge < crop_long:
                            scale = out_long_edge / float(crop_long)
                            tgt_w = max(1, int(round(src_w * cw_factor * scale)))
                            tgt_h = max(1, int(round(src_h * ch_factor * scale)))
                        else:
                            scale_pct = int(task.params.get("scale_pct", 100))
                            tgt_w = max(1, int(src_w * cw_factor * scale_pct / 100.0))
                            tgt_h = max(1, int(src_h * ch_factor * scale_pct / 100.0))
                    bit_depth = int(task.params.get("bit_depth", 8))

                    eff_lut = lut_arr
                    if eff_lut is None and hasattr(self.engine, "get_3d_lut"):
                        film_stock = task.params.get("film_stock", "kodak_portra_400")
                        paper_stock = task.params.get("paper_stock", "kodak_2383")
                        print_mode = task.params.get("print_mode", "optical")
                        try:
                            eff_lut = self.engine.get_3d_lut(film_stock, paper_stock, lut_size=33, print_mode=print_mode)
                        except Exception as ex_lut:
                            logger.warning(f"Failed to fetch 3D LUT for GPU export: {ex_lut}")

                    rendered_arr = self.canvas.render_offscreen(
                        tgt_w, tgt_h,
                        float_img_rgb=raw_target,
                        params_override=task.params,
                        bit_depth=bit_depth,
                        lut_override=eff_lut
                    )
                    QCoreApplication.processEvents()
                except Exception as ex_gl:
                    logger.warning(f"GPU offscreen render exception, falling back to CPU: {ex_gl}")
                    rendered_arr = None

                if rendered_arr is None:
                    self._start_cpu_export(task)
                    return

                if task.status == "cancelled":
                    self._finish_task(task, False, "任务已取消")
                    return

                task.progress = 65
                self.task_progress.emit(task.task_id, 65)

                # Step 3: Save to disk in background worker
                save_params = dict(task.params)
                save_params["scale_pct"] = 100  # rendered_arr is already scaled to tgt_w x tgt_h by offscreen canvas
                save_worker = _SaveImageWorker(rendered_arr, task.output_path, save_params)
                self._current_worker = save_worker

                def _on_save_done(succ, err):
                    if self._current_worker is save_worker:
                        self._current_worker = None
                    if task.status == "cancelled":
                        self._finish_task(task, False, "任务已取消")
                    elif succ:
                        self._finish_task(task, True, "")
                    else:
                        self._finish_task(task, False, err or "写入文件失败")

                save_worker.finished_save.connect(_on_save_done)
                save_worker.start()

            QTimer.singleShot(0, _do_render)

        worker.finished_load.connect(_on_load_done)
        worker.start()

    def _start_cpu_export(self, task):
        worker = _CpuExportWorker(self.engine, task)
        self._current_worker = worker

        def _on_prog(val):
            if task.status != "cancelled":
                task.progress = val
                self.task_progress.emit(task.task_id, val)

        def _on_done(succ, err):
            if self._current_worker is worker:
                self._current_worker = None
            if task.status == "cancelled":
                self._finish_task(task, False, "任务已取消")
            elif succ:
                self._finish_task(task, True, "")
            else:
                self._finish_task(task, False, err)

        worker.progress.connect(_on_prog)
        worker.finished_export.connect(_on_done)
        worker.start()

    def _finish_task(self, task, success, error_msg):
        if success:
            task.status = "finished"
            task.progress = 100
            task.error_msg = ""
        else:
            if task.status != "cancelled":
                task.status = "failed"
                task.error_msg = str(error_msg or "扫描失败")

        self.task_progress.emit(task.task_id, task.progress)
        self.queue_changed.emit()
        if self._current_task is task or self._current_task is None or getattr(self._current_task, "status", "") in ("cancelled", "finished", "failed"):
            self._current_task = None
            self._current_worker = None
        self._update_overall_status()

        # Chain immediately to next task
        QTimer.singleShot(10, self._process_next)

    def _update_overall_status(self):
        active = [t for t in self.tasks if t.status in ("queued", "running")]
        self.status_changed.emit(len(active), len(self.tasks))
        if not active:
            finished = [t for t in self.tasks if t.status == "finished"]
            if finished:
                self.status_updated.emit("所有扫描任务已完成", False)
            else:
                self.status_updated.emit("", False)
            return

        running = next((t for t in active if t.status == "running"), None)
        total_active = len(active)
        if running:
            self.status_updated.emit(f"正在扫描: {running.filename} (队列中剩余 {total_active - 1} 张)", True)
        else:
            self.status_updated.emit(f"扫描排队中 ({total_active} 项待处理)...", True)

    def close(self):
        self._is_active = False
        if self._current_worker:
            if hasattr(self._current_worker, "cancel"):
                self._current_worker.cancel()
            if hasattr(self._current_worker, "isRunning") and self._current_worker.isRunning():
                self._current_worker.wait(1000)
            if hasattr(self._current_worker, "isRunning") and self._current_worker.isRunning():
                if not hasattr(self, "_retained_workers"):
                    self._retained_workers = []
                self._retained_workers.append(self._current_worker)
            self._current_worker = None


class ExportQueueDialog(QDialog):
    """Floating Drawer Dialog for Export Queue Management."""

    def __init__(self, queue_mgr: ExportQueueManager, parent=None):
        super().__init__(parent)
        self.queue_mgr = queue_mgr
        self.setWindowTitle("扫描队列管理")
        # 单张任务卡 86px + 8px 间距，太矮只能看到 3 张。给一个能放 5 张的默认高度，
        # 同时允许往上拉和往下收（最小 460 = 约 3.5 张）。
        self.setMinimumSize(680, 460)
        self._restore_size()
        self._done_seq = 0            # 历史区的排序依据：谁最近完成，谁在最上面
        self._history_expanded = False

        try:
            from core.path_utils import get_resource_dir
            ico_path = os.path.join(get_resource_dir(), "app_icon.ico")
            png_path = os.path.join(get_resource_dir(), "app_icon.png")
            icon_file = ico_path if os.path.exists(ico_path) else png_path
            if os.path.exists(icon_file):
                self.setWindowIcon(QIcon(icon_file))
        except Exception:
            pass

        # Explicitly configure window flags to ensure WindowCloseButtonHint is 100% active on Windows
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowTitleHint |
            Qt.WindowType.WindowSystemMenuHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        is_dark = theme_manager.is_dark()
        if is_dark:
            self.setStyleSheet("""
                QDialog {
                    background: #14151a;
                    border: 1px solid #2b2e3b;
                    border-radius: 8px;
                }
                QLabel {
                    color: #e2e8f0;
                    font-family: "Microsoft YaHei UI", sans-serif;
                }
                QPushButton {
                    font-family: "Microsoft YaHei UI", sans-serif;
                    font-size: 11px;
                    padding: 4px 10px;
                    border-radius: 4px;
                }
                QPushButton:pressed {
                    padding-top: 5px;
                    padding-bottom: 3px;
                }
                QScrollArea {
                    border: none;
                    background: transparent;
                }
                QScrollBar:vertical {
                    background: rgba(20, 22, 28, 0.6);
                    width: 6px;
                    margin: 0px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical {
                    background: #3f4350;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical:hover {
                    background: #cbd5e1;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:pressed {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    border: none;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background: #ebedf2;
                    border: 1px solid #d5d9e4;
                    border-radius: 8px;
                }
                QLabel {
                    color: #1e293b;
                    font-family: "Microsoft YaHei UI", sans-serif;
                }
                QPushButton {
                    font-family: "Microsoft YaHei UI", sans-serif;
                    font-size: 11px;
                    padding: 4px 10px;
                    border-radius: 4px;
                }
                QPushButton:pressed {
                    padding-top: 5px;
                    padding-bottom: 3px;
                }
                QScrollArea {
                    border: none;
                    background: transparent;
                }
                QScrollBar:vertical {
                    background: transparent;
                    width: 6px;
                    margin: 0px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical {
                    background: #94a3b8;
                    min-height: 24px;
                    border-radius: 3px;
                    border: none;
                }
                QScrollBar::handle:vertical:hover {
                    background: #64748b;
                    border-radius: 3px;
                }
                QScrollBar::handle:vertical:pressed {
                    background: #f59e0b;
                    border-radius: 3px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                    width: 0px;
                    background: none;
                    border: none;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: none;
                    border: none;
                }
            """)

        self._cards = {}
        self._empty_label = None
        self._init_ui()

        self.queue_mgr.queue_changed.connect(self._refresh_list)
        self.queue_mgr.task_progress.connect(self._on_task_progress)
        self._refresh_list()

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_titlebar(self)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    def _init_ui(self):
        is_dark = theme_manager.is_dark()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 8, 16)
        layout.setSpacing(12)

        # Header Bar
        hdr = QHBoxLayout()
        title = QLabel("扫描队列")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f59e0b;")
        hdr.addWidget(title)
        hdr.addStretch()

        self.btn_cancel_all = QPushButton("全部取消")
        self.btn_cancel_all.setToolTip("取消所有排队中与正在扫描的任务")
        if is_dark:
            self.btn_cancel_all.setStyleSheet("""
                QPushButton {
                    background: rgba(239, 68, 68, 0.15);
                    border: 1px solid rgba(239, 68, 68, 0.40);
                    color: #f87171;
                }
                QPushButton:hover {
                    background: rgba(239, 68, 68, 0.28);
                    color: #ffffff;
                }
                QPushButton:pressed {
                    background: rgba(220, 38, 38, 0.50);
                    color: #ffffff;
                    border-color: #ef4444;
                }
            """)
        else:
            self.btn_cancel_all.setStyleSheet("""
                QPushButton {
                    background: #fee2e2;
                    border: 1px solid #fca5a5;
                    color: #dc2626;
                }
                QPushButton:hover {
                    background: #fecaca;
                    color: #b91c1c;
                }
                QPushButton:pressed {
                    background: #f87171;
                    color: #ffffff;
                    border-color: #ef4444;
                }
            """)
        self.btn_cancel_all.clicked.connect(self.queue_mgr.cancel_all)
        hdr.addWidget(self.btn_cancel_all)

        self.btn_clear_hist = QPushButton("清除已完成")
        self.btn_clear_hist.setToolTip("从列表中清除已完成或已取消的历史任务")
        if is_dark:
            self.btn_clear_hist.setStyleSheet("""
                QPushButton {
                    background: #1e2029;
                    border: 1px solid #363949;
                    color: #94a3b8;
                }
                QPushButton:hover {
                    background: #282b37;
                    color: #e2e8f0;
                }
                QPushButton:pressed {
                    background: #14151a;
                    color: #cbd5e1;
                    border-color: #282b37;
                }
            """)
        else:
            self.btn_clear_hist.setStyleSheet("""
                QPushButton {
                    background: #ffffff;
                    border: 1px solid #cbd5e1;
                    color: #475569;
                }
                QPushButton:hover {
                    background: #f1f5f9;
                    color: #1e293b;
                }
                QPushButton:pressed {
                    background: #e2e8f0;
                    color: #0f172a;
                    border-color: #94a3b8;
                }
            """)
        self.btn_clear_hist.clicked.connect(self.queue_mgr.clear_history)
        hdr.addWidget(self.btn_clear_hist)
        # 右侧外边距收到 8px 后按钮会整体右移，补一段等宽间距让表头维持原位。
        hdr.addSpacing(8)

        layout.addLayout(hdr)

        # Scrollable Task List
        self.scroll = SmoothScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        # 内容不足一屏时，容器默认会露出浅色调色板底色（暗色主题下就是一块白）。卡片
        # 自带背景样式，会盖过这里的透明声明，所以整片刷透明是安全的。
        self.container.setStyleSheet("background: transparent;")
        self.scroll.viewport().setStyleSheet("background: transparent;")
        self.item_layout = QVBoxLayout(self.container)
        # 右侧留 8px 内边距：滚动条原本紧贴卡片右缘（间隙 1px），挪进沟槽正中后
        # 卡片绝对位置不变，观感是「卡片 | 8px | 滚动条 6px | 8px | 窗口边缘」。
        self.item_layout.setContentsMargins(0, 0, 8, 0)
        self.item_layout.setSpacing(8)

        # 历史区与活跃区之间的虚线分隔：一眼看出"从这里往下都是过去"。
        self._history_sep = QFrame()
        self._history_sep.setFixedHeight(1)
        self._history_sep.setFrameShape(QFrame.Shape.NoFrame)
        self._history_sep.setStyleSheet(
            "background: transparent; border: none; border-top: 1px dashed %s;"
            % ("#4a4e5e" if is_dark else "#b3becd"))
        self._history_sep.hide()

        # 历史超过一屏时折叠成一行，点开才铺开。
        self._fold_row = QPushButton("")
        self._fold_row.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fold_row.setFixedHeight(26)
        self._fold_row.setStyleSheet(
            ("QPushButton { background: transparent; border: none; color: #94a3b8;"
             " font-size: 11px; text-align: left; padding: 0px 2px; }"
             "QPushButton:hover { color: #f59e0b; }") if is_dark else
            ("QPushButton { background: transparent; border: none; color: #64748b;"
             " font-size: 11px; text-align: left; padding: 0px 2px; }"
             "QPushButton:hover { color: #b45309; }"))
        self._fold_row.clicked.connect(self._toggle_history)
        self._fold_row.hide()

        self.item_layout.addWidget(self._history_sep)
        self.item_layout.addWidget(self._fold_row)
        self.item_layout.addStretch()
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

    # --- window size -------------------------------------------------------
    QUEUE_SIZE_PREF = "export_queue_size"

    def _restore_size(self):
        """恢复上次拖出来的尺寸；没有记录或超出屏幕就用默认 680×600。"""
        w, h = 680, 600
        try:
            from core import config_manager
            saved = config_manager.get_preferences().get(self.QUEUE_SIZE_PREF)
            if isinstance(saved, (list, tuple)) and len(saved) == 2:
                w, h = int(saved[0]), int(saved[1])
        except Exception:
            pass
        try:
            from PySide6.QtGui import QGuiApplication
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                w = max(self.minimumWidth(), min(w, max(self.minimumWidth(), avail.width() - 40)))
                h = max(self.minimumHeight(), min(h, max(self.minimumHeight(), avail.height() - 60)))
        except Exception:
            pass
        self.resize(w, h)

    def done(self, result):
        try:
            from core import config_manager
            config_manager.set_preference(self.QUEUE_SIZE_PREF, [self.width(), self.height()])
        except Exception:
            pass
        super().done(result)

    # --- ordering / folding ------------------------------------------------
    FOLD_THRESHOLD = 4

    def _toggle_history(self):
        self._history_expanded = not self._history_expanded
        self._refresh_list()

    def _ordered_tasks(self, tasks):
        """活跃 → 失败 → 历史。

        失败刻意不沉底：扫坏了的那张要留在靠上的位置，否则错误会被历史埋掉。
        历史区按完成先后倒序，紧贴活跃区的第一张永远是刚扫完的那张。
        """
        active = [t for t in tasks if t.status in ("queued", "running")]
        failed = [t for t in tasks if t.status == "failed"]
        done = [t for t in tasks if t.status in ("finished", "cancelled")]
        done.sort(key=lambda t: -int(getattr(t, "_done_seq", 0) or 0))
        return active, failed, done

    def _stage_widgets(self, active, failed, done):
        """把卡片按目标顺序放回布局；隐藏的分支只从布局摘掉，不销毁。"""
        bar = self.scroll.verticalScrollBar()
        keep = bar.value()
        at_top = keep <= 0

        fold_visible = len(done) > self.FOLD_THRESHOLD
        show_done_cards = bool(done) and (self._history_expanded or not fold_visible)

        widgets = []
        for t in active + failed:
            widgets.append(self._cards[t.task_id]["card"])
        if done:
            widgets.append(self._history_sep)
            if fold_visible:
                widgets.append(self._fold_row)
            if show_done_cards:
                for t in done:
                    widgets.append(self._cards[t.task_id]["card"])

        managed = [w for d in self._cards.values() for w in (d.get("card"),)]
        managed += [self._history_sep, self._fold_row]
        for w in managed:
            if w is not None:
                self.item_layout.removeWidget(w)
                w.hide()

        for i, w in enumerate(widgets):
            self.item_layout.insertWidget(i, w)
            w.show()

        if fold_visible:
            self._fold_row.setText(
                f"已完成 {len(done)} 条 · {'收起' if self._history_expanded else '展开'}")

        if at_top:
            bar.setValue(0)
        else:
            # 布局重新计算后滚动范围才会更新，延后一拍再还原，避免被旧的 range 截断。
            QTimer.singleShot(0, lambda v=keep: bar.setValue(min(v, bar.maximum())))

    def _on_task_progress(self, task_id, pct):
        """Lightweight live progress bar updater without triggering full card rebuilds."""
        card_item = self._cards.get(task_id)
        if card_item:
            pbar = card_item.get("pbar")
            if pbar:
                pbar.setValue(pct)

    def _refresh_list(self):
        if getattr(self, "_is_refreshing_ui", False):
            return
        self._is_refreshing_ui = True
        try:
            tasks = list(self.queue_mgr.tasks)
            if not tasks:
                for item in list(self._cards.values()):
                    btn_x = item.get("btn_x")
                    if btn_x:
                        try:
                            btn_x.clicked.disconnect()
                        except Exception:
                            pass
                    card = item.get("card")
                    if card:
                        self.item_layout.removeWidget(card)
                        card.deleteLater()
                self._cards.clear()
                self._history_sep.hide()
                self._fold_row.hide()

                if not self._empty_label:
                    self._empty_label = QLabel("当前暂无扫描任务")
                    self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._empty_label.setStyleSheet("color: #64748b; font-size: 12px; padding: 40px;")
                    self.item_layout.insertWidget(0, self._empty_label)
                self._empty_label.show()
                return

            if self._empty_label:
                self._empty_label.hide()

            task_id_set = {t.task_id for t in tasks}
            # Remove pruned items
            for old_id in list(self._cards.keys()):
                if old_id not in task_id_set:
                    item = self._cards.pop(old_id)
                    btn_x = item.get("btn_x")
                    if btn_x:
                        try:
                            btn_x.clicked.disconnect()
                        except Exception:
                            pass
                    card = item.get("card")
                    if card:
                        self.item_layout.removeWidget(card)
                        card.deleteLater()
        finally:
            self._is_refreshing_ui = False

        # 先把状态与卡片准备好，再统一排序落位。
        for task in tasks:
            if task.task_id in self._cards:
                self._update_task_card(self._cards[task.task_id], task)
            else:
                card, item_dict = self._build_task_card(task)
                self._cards[task.task_id] = item_dict

        # 记录"最近完成"的先后：历史区按它倒序，刚扫完的那张永远紧贴活跃区。
        # 用对话框自己的计数器而不是任务时间戳，避免为排序去改调度器。
        for task in tasks:
            item = self._cards[task.task_id]
            prev = item.get("status")
            if prev == task.status:
                continue
            item["status"] = task.status
            if task.status in ("finished", "cancelled", "failed") and \
                    prev not in ("finished", "cancelled", "failed"):
                self._done_seq += 1
                task._done_seq = self._done_seq

        active, failed, done = self._ordered_tasks(tasks)
        self._stage_widgets(active, failed, done)

    def _update_task_card(self, card_dict: dict, task: ExportTask):
        st_lbl = card_dict.get("st_lbl")
        pbar = card_dict.get("pbar")
        btn_x = card_dict.get("btn_x")

        status_map = {
            "queued": ("排队中", "#94a3b8"),
            "running": ("扫描中...", "#f59e0b"),
            "finished": ("已完成", "#f59e0b"),
            "cancelled": ("已取消", "#64748b"),
            "failed": (f"失败: {task.error_msg}", "#ef4444"),
        }
        st_txt, st_col = status_map.get(task.status, (task.status, "#94a3b8"))
        if st_lbl:
            st_lbl.setText(st_txt)
            st_lbl.setStyleSheet(f"font-size: 10px; color: {st_col}; font-weight: bold; background: transparent; border: none;")

        if pbar:
            pbar.setValue(task.progress if task.status != "queued" else 0)
            pbar_color = "#f59e0b" if task.status in ("running", "finished") else ("#64748b" if task.status == "cancelled" else "#ef4444")
            pbar_bg = "#20232e" if theme_manager.is_dark() else "#e2e8f0"
            pbar.setStyleSheet(f"""
                QProgressBar {{
                    background: {pbar_bg};
                    border: none;
                    border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background: {pbar_color};
                    border-radius: 2px;
                }}
            """)

        if btn_x:
            if task.status in ("queued", "running"):
                btn_x.show()
            else:
                btn_x.hide()

    def _build_task_card(self, task: ExportTask):
        is_dark = theme_manager.is_dark()
        card = QFrame()
        card.setObjectName("exportTaskCard")
        card_bg = "#171922" if is_dark else "#ffffff"
        card_border = "#262937" if is_dark else "#d5d9e4"
        card.setStyleSheet(f"""
            QFrame#exportTaskCard {{
                background: {card_bg};
                border: 1px solid {card_border};
                border-radius: 6px;
            }}
            QFrame#exportTaskCard QLabel {{
                background: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
            }}
        """)
        h = QHBoxLayout(card)
        h.setContentsMargins(12, 10, 12, 10)
        h.setSpacing(14)

        # Thumbnail with genuine original aspect ratio (enlarged container, fit aspect)
        lbl_th = QLabel()
        lbl_th.setObjectName("exportTaskThumb")
        lbl_th.setFixedSize(96, 64)
        lbl_th.setAlignment(Qt.AlignmentFlag.AlignCenter)
        th_bg = "#0f1015" if is_dark else "#f1f5f9"
        th_border = "none" if is_dark else "1px solid #e2e8f0"
        lbl_th.setStyleSheet(f"""
            QLabel#exportTaskThumb {{
                background: {th_bg};
                border: {th_border};
                border-radius: 4px;
            }}
        """)
        
        orig_pix = None
        if isinstance(task.thumb_img, QPixmap) and not task.thumb_img.isNull():
            orig_pix = task.thumb_img
        elif task.thumb_img is not None and isinstance(task.thumb_img, np.ndarray):
            h_im, w_im = task.thumb_img.shape[:2]
            qim = QImage(task.thumb_img.data, w_im, h_im, w_im * 3, QImage.Format.Format_RGB888).copy()
            orig_pix = QPixmap.fromImage(qim)

        if orig_pix and not orig_pix.isNull():
            pix = orig_pix.scaled(92, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl_th.setPixmap(pix)
        lbl_th.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(lbl_th)

        # Info & Progress
        mid = QVBoxLayout()
        mid.setSpacing(6)

        info_row = QHBoxLayout()
        fn_color = "#f1f5f9" if is_dark else "#1e293b"
        fn_lbl = QLabel(task.filename)
        fn_lbl.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {fn_color}; background: transparent; border: none;")
        info_row.addWidget(fn_lbl)
        info_row.addStretch()

        # Status text
        status_map = {
            "queued": ("排队中", "#94a3b8"),
            "running": ("扫描中...", "#f59e0b"),
            "finished": ("已完成", "#f59e0b"),
            "cancelled": ("已取消", "#64748b"),
            "failed": (f"失败: {task.error_msg}", "#ef4444"),
        }
        st_txt, st_col = status_map.get(task.status, (task.status, "#94a3b8"))
        st_lbl = QLabel(st_txt)
        st_lbl.setStyleSheet(f"font-size: 10px; color: {st_col}; font-weight: bold; background: transparent; border: none;")
        info_row.addWidget(st_lbl)
        mid.addLayout(info_row)

        # Target directory path with underline and click-to-reveal
        path_text = task.output_path or ""
        metrics = QFontMetrics(self.font())
        elided_path = metrics.elidedText(path_text, Qt.TextElideMode.ElideMiddle, 460)
        path_lbl = QLabel(elided_path)
        path_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        path_lbl.setToolTip(f"点击在文件资源管理器中定位:\n{path_text}")
        path_color = "#94a3b8" if is_dark else "#64748b"
        path_lbl.setStyleSheet(f"""
            QLabel {{
                font-size: 10px;
                color: {path_color};
                text-decoration: underline;
                background: transparent;
                border: none;
            }}
            QLabel:hover {{
                color: #f59e0b;
            }}
        """)

        def _make_reveal_handler(out_p):
            def _reveal(ev):
                if not out_p:
                    return
                import subprocess
                norm_p = os.path.abspath(out_p)
                if os.path.exists(norm_p):
                    subprocess.Popen(f'explorer /select,"{norm_p}"')
                else:
                    dir_p = os.path.dirname(norm_p)
                    if os.path.exists(dir_p):
                        os.startfile(dir_p)
            return _reveal

        path_lbl.mousePressEvent = _make_reveal_handler(task.output_path)
        mid.addWidget(path_lbl)

        # Progress bar
        pbar = QProgressBar()
        pbar.setFixedHeight(4)
        pbar.setTextVisible(False)
        pbar.setValue(task.progress if task.status != "queued" else 0)
        pbar_color = "#f59e0b" if task.status in ("running", "finished") else ("#64748b" if task.status == "cancelled" else "#ef4444")
        pbar_bg = "#20232e" if is_dark else "#e2e8f0"
        pbar.setStyleSheet(f"""
            QProgressBar {{
                background: {pbar_bg};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {pbar_color};
                border-radius: 2px;
            }}
        """)
        mid.addWidget(pbar)
        h.addLayout(mid, 1)

        # Cancel Button
        btn_x = QPushButton("✕")
        btn_x.setFixedSize(22, 22)
        btn_x.setToolTip("取消该扫描任务")
        if is_dark:
            btn_x.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #94a3b8;
                    border: 1px solid #333644;
                    border-radius: 11px;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 0;
                }
                QPushButton:hover {
                    background: rgba(239, 68, 68, 0.25);
                    color: #ef4444;
                    border-color: #ef4444;
                }
                QPushButton:pressed {
                    background: rgba(220, 38, 38, 0.50);
                    color: #ffffff;
                    border-color: #dc2626;
                }
            """)
        else:
            btn_x.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #64748b;
                    border: 1px solid #cbd5e1;
                    border-radius: 11px;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 0;
                }
                QPushButton:hover {
                    background: #fee2e2;
                    color: #ef4444;
                    border-color: #ef4444;
                }
                QPushButton:pressed {
                    background: #fca5a5;
                    color: #ffffff;
                    border-color: #dc2626;
                }
            """)
        btn_x.clicked.connect(lambda checked=False, tid=task.task_id: self.queue_mgr.cancel_task(tid))
        if task.status not in ("queued", "running"):
            btn_x.hide()
        h.addWidget(btn_x)

        item_dict = {
            "card": card,
            "st_lbl": st_lbl,
            "pbar": pbar,
            "btn_x": btn_x,
            "task_id": task.task_id
        }
        return card, item_dict
