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
        try:
            _save_rendered_file(self.rendered_arr, self.output_path, self.params_dict)
            if self._is_cancelled:
                if os.path.exists(self.output_path):
                    try:
                        os.remove(self.output_path)
                    except Exception:
                        pass
                self.finished_save.emit(False, "任务已取消")
            else:
                self.finished_save.emit(True, "")
        except Exception as e:
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

        try:
            res = self.engine.export_image(
                params_dict=self.task.params,
                output_path=self.task.output_path,
                format_type=self.task.format_type,
                quality=self.task.params.get("quality", 9),
                source_path=self.task.photo_path,
                progress_cb=_prog,
                cancel_cb=_cancel
            )
            if self._is_cancelled or self.task.status == "cancelled":
                if os.path.exists(self.task.output_path):
                    try:
                        os.remove(self.task.output_path)
                    except Exception:
                        pass
                self.finished_export.emit(False, "任务已取消")
            elif res.get("success"):
                self.finished_export.emit(True, "")
            else:
                err = str(res.get("error") or "冲印显影失败")
                self.finished_export.emit(False, err)
        except Exception as e:
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
                        self._current_task = None
                        self._current_worker = None
                    self.task_progress.emit(task.task_id, task.progress)
                    self.queue_changed.emit()
                    self._update_overall_status()
                break
        if cancelled_current:
            QTimer.singleShot(20, self._process_next)

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
            self._current_task = None
            self._current_worker = None
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
                    scale_pct = int(task.params.get("scale_pct", 100))
                    tgt_w = max(1, int(src_w * scale_pct / 100.0))
                    tgt_h = max(1, int(src_h * scale_pct / 100.0))
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
                save_worker = _SaveImageWorker(rendered_arr, task.output_path, task.params)
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
                task.error_msg = str(error_msg or "冲印失败")

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
                self.status_updated.emit("所有冲印任务已完成", False)
            else:
                self.status_updated.emit("", False)
            return

        running = next((t for t in active if t.status == "running"), None)
        total_active = len(active)
        if running:
            self.status_updated.emit(f"正在冲印: {running.filename} (队列中剩余 {total_active - 1} 张)", True)
        else:
            self.status_updated.emit(f"冲印排队中 ({total_active} 项待处理)...", True)

    def close(self):
        self._is_active = False
        if self._current_worker:
            if hasattr(self._current_worker, "cancel"):
                self._current_worker.cancel()
            self._current_worker.wait(500)


class ExportQueueDialog(QDialog):
    """Floating Drawer Dialog for Export Queue Management."""

    def __init__(self, queue_mgr: ExportQueueManager, parent=None):
        super().__init__(parent)
        self.queue_mgr = queue_mgr
        self.setWindowTitle("冲印队列管理")
        self.setFixedSize(680, 420)

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
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header Bar
        hdr = QHBoxLayout()
        title = QLabel("冲印队列")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #f59e0b;")
        hdr.addWidget(title)
        hdr.addStretch()

        self.btn_cancel_all = QPushButton("全部取消")
        self.btn_cancel_all.setToolTip("取消所有排队中与正在冲印的任务")
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

        layout.addLayout(hdr)

        # Scrollable Task List
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.item_layout = QVBoxLayout(self.container)
        self.item_layout.setContentsMargins(0, 0, 0, 0)
        self.item_layout.setSpacing(8)
        self.item_layout.addStretch()
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

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

                if not self._empty_label:
                    self._empty_label = QLabel("当前暂无冲印任务")
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

        # Maintain chronological forward order (earliest task at top)
        for idx, task in enumerate(tasks):
            if task.task_id in self._cards:
                self._update_task_card(self._cards[task.task_id], task)
            else:
                card, item_dict = self._build_task_card(task)
                self._cards[task.task_id] = item_dict
                insert_idx = max(0, self.item_layout.count() - 1)
                self.item_layout.insertWidget(insert_idx, card)

    def _update_task_card(self, card_dict: dict, task: ExportTask):
        st_lbl = card_dict.get("st_lbl")
        pbar = card_dict.get("pbar")
        btn_x = card_dict.get("btn_x")

        status_map = {
            "queued": ("排队中", "#94a3b8"),
            "running": ("冲印中...", "#f59e0b"),
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
            "running": ("冲印中...", "#f59e0b"),
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
        btn_x.setToolTip("取消该冲印任务")
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
