# src/dlc_live_torch/app.py

import sys
import os
import time
import cv2
import numpy as np
from collections import deque
import multiprocessing as mp
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QHBoxLayout, QVBoxLayout, QFileDialog, QSizePolicy, QMessageBox, QSlider,
    QGroupBox, QFormLayout, QRadioButton, QLineEdit, QCheckBox, QToolBar, QAction
)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap

# Conditional import for psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
# Local imports from our new package structure
from .config import load_app_config, save_app_config
from .widgets import CroppableLabel, CropManager, ActiveQueueReference
from .workers import InferenceProcessManager, VideoProcessingThread, GuiUpdateWorker


class App(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("DLC Live - PyTorch GUI")
        self.settings = {}; self.video_processing_thread = None; self.inference_manager_thread = None
        self.gui_update_worker = None; self.video_writer = None; self.latest_original_frame = None
        self.latest_processed_frame = None;
        
        self.main_shutdown_event = mp.Event()
        self.inference_shutdown_event = mp.Event()
        
        self.results_queue = mp.Queue(maxsize=10); self.preview_queue = mp.Queue(maxsize=2)
        self.active_queue_ref = ActiveQueueReference(); self.camera_command_queue = mp.Queue()
        self.display_fps_tracker = deque(maxlen=30)
        
        self.frames_captured_counter = mp.Value('i', 0); self.frames_enqueued_counter = mp.Value('i', 0)
        self.frames_processed_counter = mp.Value('i', 0); self.csv_write_counter = mp.Value('i', 0)
        self.stats_timer = QTimer(self); self.stats_timer.setInterval(500); self.stats_timer.timeout.connect(self.update_stats_display)
        self.last_csv_count = 0; self.last_stats_time = time.monotonic()
        self.crop_manager = CropManager()
        self.app_config = load_app_config()
        
        self.is_recording = False
        self.recording_buffer = []
        self.record_w = 640; self.record_h = 360; self.record_fps = 30
        
        self.create_widgets()
        self.update_camera_source_ui()
        self.apply_loaded_config()
        
        self.start_preview_threads()

    def start_preview_threads(self):
        print("Starting preview threads (VideoProcessing + GuiUpdate)...")
        if not self.gather_settings(starting_analysis=False):
            QMessageBox.warning(self, "Config Error", "Could not load default settings. Camera may not start.")
        
        try:
            cam_w = int(self.w_edit.text())
            cam_h = int(self.h_edit.text())
            aspect = cam_w / float(cam_h) if cam_h > 0 else 16.0/9.0
            self.preview_width = 640
            self.preview_height = int(self.preview_width / aspect)
        except Exception:
            self.preview_width = 640
            self.preview_height = 360
        
        print(f"Preview size set to: {self.preview_width}x{self.preview_height}")
        self.settings['preview_width_actual'] = self.preview_width
        self.settings['preview_height_actual'] = self.preview_height
        
        self.video_processing_thread = VideoProcessingThread(self.settings, self.active_queue_ref,
                                                             self.preview_queue, self.main_shutdown_event,
                                                             self.camera_command_queue, self.frames_captured_counter,
                                                             self.frames_enqueued_counter, self.crop_manager)
        self.gui_update_worker = GuiUpdateWorker(self.results_queue, self.preview_queue,
                                                 self.main_shutdown_event, self.settings.get('display_fps', 60))
        self.gui_update_worker.new_frame_ready.connect(self.update_video_feed)
        
        self.video_processing_thread.start()
        self.gui_update_worker.start()

    def create_widgets(self):
        self.create_toolbar(); mw = QWidget(); self.setCentralWidget(mw); layout = QHBoxLayout(mw)
        ctrl_layout = QVBoxLayout(); vid_layout = QVBoxLayout(); layout.addLayout(ctrl_layout); layout.addLayout(vid_layout, 1)

        g = QGroupBox("1. Input"); ctrl_layout.addWidget(g); f = QFormLayout(g)
        self.cfg_edit = self._create_file_input(f, "Config:"); self.snap_edit = self._create_file_input(f, "Snapshot:")
        g = QGroupBox("2. Source"); ctrl_layout.addWidget(g); v = QVBoxLayout(g)
        self.usb_rb = QRadioButton("USB"); self.ip_rb = QRadioButton("IP Cam"); self.file_rb = QRadioButton("File")
        for rb in [self.usb_rb, self.ip_rb, self.file_rb]: v.addWidget(rb); rb.toggled.connect(self.update_camera_source_ui); rb.toggled.connect(self.save_current_config)
        h = QHBoxLayout(); self.src_lbl = QLabel("Index:"); self.src_edit = QLineEdit(); self.src_btn = QPushButton("...")
        self.src_btn.clicked.connect(lambda: self.browse_file(self.src_edit)); h.addWidget(self.src_lbl); h.addWidget(self.src_edit); h.addWidget(self.src_btn); v.addLayout(h)
        self.src_edit.textChanged.connect(self.save_current_config)
        self.cam_grp = QGroupBox("3. Cam Params"); ctrl_layout.addWidget(self.cam_grp); f = QFormLayout(self.cam_grp)
        self.w_edit = QLineEdit(); self.h_edit = QLineEdit(); self.fps_edit = QLineEdit()
        f.addRow("W:", self.w_edit); f.addRow("H:", self.h_edit); f.addRow("FPS:", self.fps_edit)
        for edit in [self.w_edit, self.h_edit, self.fps_edit]: edit.textChanged.connect(self.save_current_config)
        g = QGroupBox("4. Crop"); ctrl_layout.addWidget(g); f = QFormLayout(g)
        self.crop_cb = QCheckBox("Enable Crop"); self.crop_cb.toggled.connect(self.toggle_cropping_mode)
        self.crop_rst_btn = QPushButton("Reset"); self.crop_rst_btn.clicked.connect(self.reset_crop); self.crop_lbl = QLabel("N/A")
        f.addRow(self.crop_cb); f.addRow(self.crop_rst_btn); f.addRow("Current:", self.crop_lbl)
        self.rt_grp = QGroupBox("5. Realtime"); ctrl_layout.addWidget(self.rt_grp); f = QFormLayout(self.rt_grp)
        self.exp_sld = self._create_slider(f, "Exp:", -13, -1, -11, self.update_exposure); self.gain_sld = self._create_slider(f, "Gain:", 0, 128, 0, self.update_gain); self.wb_sld = self._create_slider(f, "WB:", 2000, 8000, 4000, self.update_white_balance)
        for sld in [self.exp_sld, self.gain_sld, self.wb_sld]: sld.valueChanged.connect(self.save_current_config)
        g = QGroupBox("6. Preproc"); ctrl_layout.addWidget(g); f = QFormLayout(g)
        self.pre_cmb = QComboBox(); self.pre_cmb.addItems(["None", "Flat-field", "Morph Open"]); self.pre_cmb.currentTextChanged.connect(self.update_preprocessing_ui); self.pre_cmb.currentTextChanged.connect(self.save_current_config); f.addRow("Method:", self.pre_cmb)
        self.flat_edit = self._create_file_input(f, "Flat Img:"); self.flat_edit.parent().setVisible(False)
        g = QGroupBox("7. Perf"); ctrl_layout.addWidget(g); f = QFormLayout(g)
        self.ram_sld = self._create_slider(f, "RAM Rst:", 4, 32, 16); self.ram_sld.valueChanged.connect(self.save_current_config)
        if not PSUTIL_AVAILABLE: self.ram_sld.setEnabled(False); f.addRow(QLabel("<i>psutil N/A</i>"))
        self.bs_sld = self._create_slider(f, "Batch(1):", 1, 1, 1); self.bs_sld.setEnabled(False)
        self.fp16_cb = QCheckBox("Use FP16"); self.fp16_cb.stateChanged.connect(self.save_current_config); f.addRow(self.fp16_cb)
        self.disp_fps_sld = self._create_slider(f, "Disp FPS:", 1, 120, 60); self.disp_fps_sld.valueChanged.connect(self.save_current_config)
        self.skel_sld = self._create_slider(f, "Skel Conf:", 0, 100, 10); self.skel_sld.valueChanged.connect(self.save_current_config)
        self.pt_sld = self._create_slider(f, "Pt Conf:", 0, 100, 60); self.pt_sld.valueChanged.connect(self.save_current_config)
        self.show_skel_cb = QCheckBox("Show Skel"); self.show_skel_cb.stateChanged.connect(self.save_current_config); f.addRow(self.show_skel_cb)
        
        g = QGroupBox("8. Output"); ctrl_layout.addWidget(g); f = QFormLayout(g)
        self.csv_cb = QCheckBox("Save CSV"); self.csv_cb.stateChanged.connect(self.save_current_config); self.csv_edit = self._create_file_input(f, self.csv_cb, False)
        
        g = QGroupBox("9. Capture"); ctrl_layout.addWidget(g); h = QHBoxLayout(g); self.pht_btn = QPushButton("Photo"); self.pht_btn.clicked.connect(self.take_photo); self.rec_btn = QPushButton("Record"); self.rec_btn.setCheckable(True); self.rec_btn.clicked.connect(self.toggle_recording)
        h.addWidget(self.pht_btn); h.addWidget(self.rec_btn)
        
        g = QGroupBox("10. Stats"); ctrl_layout.addWidget(g); v = QVBoxLayout(g); self.st_lbl = QLabel("N/A"); self.st_lbl.setAlignment(Qt.AlignLeft); v.addWidget(self.st_lbl)
        
        self.start_btn = QPushButton("Start Analysis"); self.start_btn.clicked.connect(self.start_analysis);
        self.stop_btn = QPushButton("Stop Analysis"); self.stop_btn.clicked.connect(self.stop_analysis); self.stop_btn.setEnabled(False)
        h = QHBoxLayout(); h.addWidget(self.start_btn); h.addWidget(self.stop_btn); ctrl_layout.addLayout(h); ctrl_layout.addStretch()

        g = QGroupBox('Preview'); vid_layout.addWidget(g, 1); h = QHBoxLayout(g)
        self.orig_lbl = CroppableLabel('Original'); self.orig_lbl.crop_finished.connect(self.on_crop_finished)
        self.proc_lbl = QLabel('Processed')
        for lbl in [self.orig_lbl, self.proc_lbl]: lbl.setAlignment(Qt.AlignCenter); lbl.setMinimumSize(320, 240); lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); h.addWidget(lbl)
        self.lat_lbl = QLabel("Inf: -- | Cap-CSV: -- | E2E: -- | RAM: --"); self.statusBar().addPermanentWidget(self.lat_lbl)

    def apply_loaded_config(self):
        print("Applying loaded configuration to GUI...")
        for widget in self.findChildren(QWidget):
            if hasattr(widget, 'blockSignals'): widget.blockSignals(True)
        try:
            self.cfg_edit.setText(self.app_config.get('dlc_config_path', ''))
            self.snap_edit.setText(self.app_config.get('snapshot_path', ''))
            source_type = self.app_config.get('camera_source', 'USB Webcam')
            if source_type == 'USB Webcam': self.usb_rb.setChecked(True)
            elif source_type == 'IP Camera': self.ip_rb.setChecked(True)
            elif source_type == 'Video File': self.file_rb.setChecked(True)
            self.src_edit.setText(self.app_config.get('camera_path', '0'))
            self.update_camera_source_ui()
            self.w_edit.setText(str(self.app_config.get('cam_width', 640)))
            self.h_edit.setText(str(self.app_config.get('cam_height', 480)))
            self.fps_edit.setText(str(self.app_config.get('cam_fps', 60)))
            self.exp_sld.setValue(self.app_config.get('exposure', -11))
            self.gain_sld.setValue(self.app_config.get('gain', 0))
            self.wb_sld.setValue(self.app_config.get('white_balance', 4000))
            self.pre_cmb.setCurrentText(self.app_config.get('preproc_method', 'None'))
            self.flat_edit.setText(self.app_config.get('flat_image_path', ''))
            self.update_preprocessing_ui(self.pre_cmb.currentText())
            self.ram_sld.setValue(self.app_config.get('ram_threshold_gb', 16))
            self.bs_sld.setValue(1)
            self.fp16_cb.setChecked(self.app_config.get('use_fp16', True))
            self.disp_fps_sld.setValue(self.app_config.get('display_fps', 60))
            self.skel_sld.setValue(int(self.app_config.get('skeleton_confidence', 0.10) * 100))
            self.pt_sld.setValue(int(self.app_config.get('point_confidence', 0.60) * 100))
            self.show_skel_cb.setChecked(self.app_config.get('show_skeleton', True))
            self.csv_cb.setChecked(self.app_config.get('save_csv', False))
            self.csv_edit.setText(self.app_config.get('csv_output_path', ''))
            crop_coords = self.app_config.get('crop_coords', None)
            if crop_coords:
                self.crop_manager.set_from_tuple(crop_coords)
                x1,y1,x2,y2 = crop_coords
                self.crop_lbl.setText(f"X:{x1}-{x2},Y:{y1}-{y2}")
            else:
                self.crop_manager.clear(); self.crop_lbl.setText("N/A")
        finally:
            for widget in self.findChildren(QWidget):
                if hasattr(widget, 'blockSignals'): widget.blockSignals(False)

    def save_current_config(self):
        if not hasattr(self, 'cfg_edit'): return
        crop_tuple = self.crop_manager.get_tuple()
        crop_to_save = list(crop_tuple) if crop_tuple else None
        
        current_config = {
            'dlc_config_path': self.cfg_edit.text(),
            'snapshot_path': self.snap_edit.text(),
            'pytorch_config_path': self.derive_pytorch_config_path(self.snap_edit.text()),
            'camera_source': 'USB Webcam' if self.usb_rb.isChecked() else ('IP Camera' if self.ip_rb.isChecked() else 'Video File'),
            'camera_path': self.src_edit.text(),
            'cam_width': int(self.w_edit.text()) if self.w_edit.text().isdigit() else 640,
            'cam_height': int(self.h_edit.text()) if self.h_edit.text().isdigit() else 480,
            'cam_fps': int(self.fps_edit.text()) if self.fps_edit.text().isdigit() else 60,
            'exposure': self.exp_sld.value(), 'gain': self.gain_sld.value(), 'white_balance': self.wb_sld.value(),
            'preproc_method': self.pre_cmb.currentText(), 'flat_image_path': self.flat_edit.text(),
            'use_fp16': self.fp16_cb.isChecked(), 'display_fps': self.disp_fps_sld.value(), 'ram_threshold_gb': self.ram_sld.value(),
            'skeleton_confidence': self.skel_sld.value() / 100.0, 'point_confidence': self.pt_sld.value() / 100.0, 'show_skeleton': self.show_skel_cb.isChecked(),
            'save_csv': self.csv_cb.isChecked(), 'csv_output_path': self.csv_edit.text(),
            'crop_coords': crop_to_save
        }
        save_app_config(current_config)

    def update_stats_display(self):
        if not self.start_btn.isEnabled() and not self.stop_btn.isEnabled(): return
        now = time.monotonic(); elapsed = now - self.last_stats_time
        if elapsed < 0.1: return
        cap=self.frames_captured_counter.value; enq=self.frames_enqueued_counter.value
        proc=self.frames_processed_counter.value; drop=cap-enq
        csv_now = self.csv_write_counter.value; csv_diff = csv_now - self.last_csv_count
        csv_fps = csv_diff / elapsed if elapsed > 0 else 0.0
        txt = (f"Cap:{cap} | Enq:{enq} | Proc:{proc} | Drop(Ovr):{drop} | CSV FPS: {csv_fps:.1f}")
        self.st_lbl.setText(txt)
        self.last_csv_count = csv_now; self.last_stats_time = now

    def toggle_cropping_mode(self, checked):
        self.orig_lbl.setEnabled(checked)
        if checked: QMessageBox.information(self,"Crop On","Drag on 'Original' feed.")
        if not checked: self.reset_crop()
        self.save_current_config()

    def reset_crop(self):
        self.crop_manager.clear(); self.orig_lbl.clear_crop(); self.crop_lbl.setText("N/A")
        print("Crop reset."); self.save_current_config()

    def on_crop_finished(self, crop_qrect):
        if self.latest_original_frame is None or crop_qrect.width()<10 or crop_qrect.height()<10:
            self.reset_crop(); QMessageBox.warning(self, "Crop Err", "No frame or crop too small."); return
        lbl_sz=self.orig_lbl.size(); pxm=self.orig_lbl.pixmap()
        if not pxm or pxm.isNull(): self.reset_crop(); return
        
        frame_h, frame_w, _ = self.latest_original_frame.shape
        
        pxm_sz=pxm.size();
        if pxm_sz.height() <= 0 or lbl_sz.height() <= 0: self.reset_crop(); return
        
        pxm_r=pxm_sz.width()/float(pxm_sz.height()); lbl_r=lbl_sz.width()/float(lbl_sz.height())
        if pxm_r==0 or lbl_r==0: self.reset_crop(); return

        if pxm_r > lbl_r: dw=lbl_sz.width(); dh=dw/pxm_r; ox=0; oy=(lbl_sz.height()-dh)/2
        else: dh=lbl_sz.height(); dw=dh*pxm_r; oy=0; ox=(lbl_sz.width()-dw)/2
        if dw<=0 or dh<=0: self.reset_crop(); return

        preview_h, preview_w = pxm_sz.height(), pxm_sz.width()
        if preview_w <= 0 or preview_h <= 0: self.reset_crop(); return

        scale_w = frame_w / float(preview_w); scale_h = frame_h / float(preview_h)
        
        pxm_x = max(0, crop_qrect.left() - ox); pxm_y = max(0, crop_qrect.top() - oy)
        pxm_x2 = min(dw, crop_qrect.right() - ox); pxm_y2 = min(dh, crop_qrect.bottom() - oy)

        x1=int(pxm_x * scale_w); y1=int(pxm_y * scale_h)
        x2=int(pxm_x2 * scale_w); y2=int(pxm_y2 * scale_h)

        x1,x2=sorted([x1,x2]); y1,y2=sorted([y1,y2])
        if (x2-x1)<10 or (y2-y1)<10: self.reset_crop(); QMessageBox.warning(self,"Crop Err","Scaled region too small."); return
        coords=(x1,y1,x2,y2); self.crop_manager.set(coords); self.crop_lbl.setText(f"X:{x1}-{x2},Y:{y1}-{y2}")
        
        label_rect = QRect(QPoint(int(pxm_x+ox), int(pxm_y+oy)), QPoint(int(pxm_x2+ox), int(pxm_y2+oy)))
        self.orig_lbl.set_existing_crop(label_rect);
        print(f"Crop set: {coords}")
        self.save_current_config()

    def create_toolbar(self):
        tb=self.addToolBar('File')
        load_act = QAction('Load Cfg', self); load_act.triggered.connect(self.load_config_from_file_dialog) ; tb.addAction(load_act)
        save_act = QAction('Save Cfg As...', self); save_act.triggered.connect(self.save_config_as_file_dialog) ; tb.addAction(save_act)
        tb.addSeparator()
        o=QAction('Flat',self); o.triggered.connect(lambda: self.browse_file(self.flat_edit)); tb.addAction(o)
        e=QAction('Exit',self); e.triggered.connect(self.close); tb.addAction(e)

    def load_config_from_file_dialog(self):
        from .config import CONFIG_FILE, load_app_config
        path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "YAML Files (*.yaml *.yml)")
        if path:
            CONFIG_FILE = path
            self.app_config = load_app_config()
            self.apply_loaded_config()
            QMessageBox.information(self, "Config Loaded", f"Loaded settings from {path}")

    def save_config_as_file_dialog(self):
        from .config import CONFIG_FILE
        path, _ = QFileDialog.getSaveFileName(self, "Save Configuration As", "", "YAML Files (*.yaml *.yml)")
        if path:
            if not (path.endswith(".yaml") or path.endswith(".yml")): path += ".yaml"
            CONFIG_FILE = path
            self.save_current_config()
            QMessageBox.information(self, "Config Saved", f"Saved current settings to {path}")

    def _create_file_input(self, layout, lbl_w, is_open=True):
        edit=QLineEdit(); btn_txt="..." if is_open else "Save"; btn=QPushButton(btn_txt)
        edit.textChanged.connect(self.save_current_config)
        btn.clicked.connect(lambda: self.browse_file(edit, is_open)); h=QHBoxLayout(); h.addWidget(edit); h.addWidget(btn)
        if isinstance(lbl_w,str):
            layout.addRow(lbl_w, h)
        else:
            layout.addRow(lbl_w, h)
        return edit

    def _create_slider(self, layout, lbl, min_v, max_v, def_v, func=None):
        sld=QSlider(Qt.Horizontal); sld.setRange(min_v,max_v); sld.setValue(def_v)
        v_lbl=QLabel(f"{def_v}"); sld.valueChanged.connect(lambda v, l=v_lbl: l.setText(f"{v}"))
        if func: sld.valueChanged.connect(func)
        sld.valueChanged.connect(self.save_current_config)
        h=QHBoxLayout(); h.addWidget(sld); h.addWidget(v_lbl); layout.addRow(lbl, h); return sld

    def browse_file(self, edit, is_open=True):
        f = QFileDialog.getOpenFileName if is_open else QFileDialog.getSaveFileName
        path, _ = f(self,"Select File")
        if path: edit.setText(path)

    def update_camera_source_ui(self):
        usb=self.usb_rb.isChecked(); ip=self.ip_rb.isChecked(); file=self.file_rb.isChecked()
        lbl="Idx:" if usb else("URL:" if ip else "Path:");
        txt = self.src_edit.text()
        if (usb or ip) and not txt: txt = "0" if usb else "rtsp://"
        self.src_lbl.setText(lbl); self.src_edit.setText(txt); self.src_btn.setVisible(file)
        self.cam_grp.setEnabled(usb); self.rt_grp.setEnabled(usb)

    def update_preprocessing_ui(self, m): self.flat_edit.parent().setVisible(m=="Flat-field")
    def update_exposure(self,v): self.camera_command_queue.put({'property':cv2.CAP_PROP_EXPOSURE,'value':float(v)})
    def update_gain(self,v): self.camera_command_queue.put({'property':cv2.CAP_PROP_GAIN,'value':float(v)})
    def update_white_balance(self,v): self.camera_command_queue.put({'property':cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,'value':float(v)})

    def gather_settings(self, starting_analysis=True):
        try:
            s={'config_path': self.cfg_edit.text(), 'pytorch_config_path': self.derive_pytorch_config_path(self.snap_edit.text()),
               'snapshot_path': self.snap_edit.text(), 'camera_source': 'USB Webcam' if self.usb_rb.isChecked() else ('IP Camera' if self.ip_rb.isChecked() else 'Video File'),
               'camera_path': self.src_edit.text(), 'batch_size': 1,
               'use_fp16': self.fp16_cb.isChecked(), 'target_fps': self.disp_fps_sld.value(),
               'skeleton_confidence': self.skel_sld.value()/100., 'point_confidence': self.pt_sld.value()/100.,
               'show_skeleton': self.show_skel_cb.isChecked(), 'cam_width': int(self.w_edit.text()), 'cam_height': int(self.h_edit.text()),
               'cam_fps': int(self.fps_edit.text()), 'method': self.pre_cmb.currentText(), 'flat_image_path': self.flat_edit.text(),
               'save_csv': self.csv_cb.isChecked(), 'csv_output_path': self.csv_edit.text(),
               'exposure': self.exp_sld.value(), 'gain': self.gain_sld.value(),
               'white_balance': self.wb_sld.value(), 'ram_threshold_gb': self.ram_sld.value()}
            
            if starting_analysis and (not s['config_path'] or not s['snapshot_path'] or not s['pytorch_config_path']):
                QMessageBox.critical(self,"Err","DLC Config, Snapshot, and Pytorch Config paths are required."); return False
                
            self.bs_sld.setValue(1); self.settings = s; return True
        except Exception as e:
            if starting_analysis: QMessageBox.critical(self,"Err",f"Settings invalid: {e}");
            else: print(f"Warn: Settings invalid on preview start: {e}")
            return False

    def derive_pytorch_config_path(self, p): return os.path.join(os.path.dirname(p),'pytorch_config.yaml') if p else ""

    def start_analysis(self):
        print("Start Analysis clicked...")
        self.save_current_config()
        if not self.gather_settings(starting_analysis=True): return
        
        self.frames_captured_counter.value=0; self.frames_enqueued_counter.value=0;
        self.frames_processed_counter.value=0; self.csv_write_counter.value = 0
        self.last_csv_count = 0; self.last_stats_time = time.monotonic()

        self.stats_timer.start(); self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.crop_cb.setEnabled(False); self.crop_rst_btn.setEnabled(False);
        self.statusBar().showMessage("Starting Inference Process...")
        
        self.inference_shutdown_event.clear()
        
        self.inference_manager_thread = InferenceProcessManager(self.settings, self.active_queue_ref, self.results_queue, self.inference_shutdown_event, self.frames_processed_counter, self.csv_write_counter)
        self.inference_manager_thread.start()
        print("Inference manager started.")

    def stop_analysis(self):
        print("Stop Analysis clicked...")
        self.stats_timer.stop()
        if not self.inference_manager_thread or not self.inference_manager_thread.is_alive():
            print("Stop analysis: Inference manager not running.")
        else:
            print("Stopping inference process..."); self.statusBar().showMessage("Shutting down inference...")
            
            if self.inference_manager_thread and self.inference_manager_thread.is_alive():
                print("Setting inference shutdown event...")
                self.inference_shutdown_event.set()
                print(f"Joining Inference Manager thread...")
                self.inference_manager_thread.join(timeout=7)
                if self.inference_manager_thread.is_alive(): print("WARN: Mgr thread no join.")
            
        self.active_queue_ref.set(None)
        
        print("Resetting UI after stopping analysis.");
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.crop_cb.setEnabled(True); self.crop_rst_btn.setEnabled(True)
        self.statusBar().showMessage("Preview running. Inference stopped.")
        self.proc_lbl.setText("Stopped."); self.lat_lbl.setText("Inf: -- | Cap-CSV: -- | E2E: -- | RAM: --")
        print("Stop analysis complete.")

    def take_photo(self):
        if self.latest_original_frame is None: QMessageBox.warning(self,"Err","No frame."); return
        ts=datetime.now().strftime('%y%m%d_%H%M%S'); o=f'orig_{ts}.jpg'; p=f'proc_{ts}.jpg'
        try:
            cv2.imwrite(o, self.latest_original_frame); msg=f"Saved: {o}"
            if self.latest_processed_frame is not None: cv2.imwrite(p, self.latest_processed_frame); msg+=f"\n- {p}"
            QMessageBox.information(self,"Saved", msg)
        except Exception as e: QMessageBox.critical(self,"Err", f"Save fail: {e}")

    def toggle_recording(self):
        if not self.is_recording:
            try:
                self.record_w = self.settings.get('preview_width_actual', 640)
                self.record_h = self.settings.get('preview_height_actual', 360)
                self.record_fps = float(self.disp_fps_sld.value())
                if self.record_w <= 0 or self.record_h <= 0 or self.record_fps <= 0:
                    raise ValueError("Invalid dimensions or FPS from settings")
                size = (self.record_w, self.record_h)
            except Exception as e:
                QMessageBox.warning(self,"Err",f"Invalid Camera/Display Parameters for recording: {e}"); return

            self.recording_buffer = []
            self.is_recording = True
            self.rec_btn.setText("Stop Rec")
            self.rec_btn.setStyleSheet("background-color: red; color: white;")
            self.statusBar().showMessage("Recording...")
            print(f"Recording started. Target size: {size} @ {self.record_fps} FPS")
        else:
            self.is_recording = False
            self.rec_btn.setText("Record")
            self.rec_btn.setStyleSheet("")
            self.statusBar().showMessage("Recording stopped. Saving video...")
            print("Recording stopped. Saving...")
            
            self.save_recorded_video()

    def save_recorded_video(self):
        if not self.recording_buffer:
            QMessageBox.warning(self,"Save Error", "No frames were recorded."); return
            
        size = (self.record_w, self.record_h)
        fps = self.record_fps
        path, _ = QFileDialog.getSaveFileName(self, "Save Recorded Video As", "", "MP4 Video (*.mp4)")
        
        if not path:
            QMessageBox.warning(self,"Save Cancelled", "Video was not saved.");
            self.recording_buffer = []
            return

        if not (path.endswith(".mp4") or path.endswith(".avi")):
            path += ".mp4"
            
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(path, fourcc, fps, size)
            if not writer.isOpened(): raise IOError("Could not open video writer.")
            
            print(f"Writing {len(self.recording_buffer)} frames to {path} at {fps} FPS, size {size}...")
            for frame in self.recording_buffer:
                if frame.shape[0] != self.record_h or frame.shape[1] != self.record_w:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
                writer.write(frame)
                
            writer.release()
            self.recording_buffer = []
            self.statusBar().showMessage(f"Video saved to {path}", 5000)
            QMessageBox.information(self,"Save Complete", f"Video saved to {path}")
            print(f"Video saved.")

        except Exception as e:
            QMessageBox.critical(self,"Save Error", f"Failed to save video: {e}")
            print(f"!!! Error saving video: {e}")
            self.recording_buffer = []

    @pyqtSlot(dict)
    def update_video_feed(self, packet):
        if self.main_shutdown_event.is_set(): return
        
        packet_type = packet.get('type')
        
        preview_frame = packet.get('preview_frame')
        if preview_frame is None:
            preview_frame = packet.get('original_frame')
            if preview_frame is None: return
            
        if packet.get('original_frame') is not None:
            self.latest_original_frame = packet['original_frame']

        if packet_type == 'processed':
            ts = packet.get('timestamps', {}); ts['display_start'] = time.monotonic()
            lat = {}
            lat['PreProc'] = (ts.get('processed', 0) - ts.get('capture', 0)) * 1000
            lat['InfQWait'] = (ts.get('dequeued_for_inference', 0) - ts.get('enqueued_for_inference', 0)) * 1000 if ts.get('enqueued_for_inference', -1)>0 else 0
            lat['Infer'] = packet.get('inference_time_ms', 0)
            lat['GUIQWait'] = (ts.get('dequeued_for_gui', 0) - ts.get('enqueued_for_gui', 0)) * 1000 if ts.get('dequeued_for_gui', -1)>0 else 0
            lat['GUIUpdate'] = (ts.get('display_start', 0) - ts.get('dequeued_for_gui', 0)) * 1000 if ts.get('dequeued_for_gui', -1)>0 else 0
            lat['CapToCSV'] = packet.get('capture_to_csv_ms', 0)
            lat['E2E'] = (ts.get('display_start', 0) - ts.get('capture', 0)) * 1000

            ram = psutil.Process(os.getpid()).memory_info().rss/(1024**3) if PSUTIL_AVAILABLE else -1
            self.lat_lbl.setText(f"Inf: {lat['Infer']:.1f} | Cap-CSV: {lat['CapToCSV']:.1f} | E2E: {lat['E2E']:.1f} | RAM: {ram:.2f} GB")

            proc=packet['processed_frame']; crop=packet.get('crop_coords')
            disp = preview_frame.copy()

            pred=packet.get('predictions')
            if pred and self.settings['show_skeleton']:
                proc_ann = proc.copy()
                # A more robust way to get skeleton might be needed here from config
                # For your finger tracking project, this is perfect.
                skel=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)] # 7 keypoints -> 6 connections
                parts = pred.get("bodyparts", [])
                if parts is not None and parts.size > 0:
                    kps_to_draw = []
                    if parts.ndim == 3: kps_to_draw = parts[0] # Take first instance for display
                    elif parts.ndim == 2: kps_to_draw = parts

                    if len(kps_to_draw) > 0:
                        for i, j in skel:
                            if max(i,j)<len(kps_to_draw):
                                p1,p2=kps_to_draw[i],kps_to_draw[j]
                                if isinstance(p1,(np.ndarray,list,tuple)) and len(p1)==3 and isinstance(p2,(np.ndarray,list,tuple)) and len(p2)==3:
                                    if p1[2]>self.settings['skeleton_confidence'] and p2[2]>self.settings['skeleton_confidence']:
                                        pt1,pt2=(int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])); cv2.line(proc_ann,pt1,pt2,(255,100,0),2)
                        for kp_data in kps_to_draw:
                            if isinstance(kp_data,(np.ndarray,list,tuple)) and len(kp_data)==3:
                                x,y,c = kp_data
                                if c>self.settings['point_confidence']: cv2.circle(proc_ann,(int(x),int(y)),5,(0,255,0),-1)

                if crop:
                    if packet.get('original_frame') is None: return
                    orig_h, orig_w = packet['original_frame'].shape[:2]
                    prev_h, prev_w = disp.shape[:2]
                    if orig_w > 0 and orig_h > 0:
                        scale_w = prev_w / float(orig_w); scale_h = prev_h / float(orig_h)
                        disp_x1 = int(crop[0] * scale_w); disp_y1 = int(crop[1] * scale_h)
                        disp_x2 = int(crop[2] * scale_w); disp_y2 = int(crop[3] * scale_h)
                        try:
                            rect_w = disp_x2 - disp_x1; rect_h = disp_y2 - disp_y1
                            if rect_w > 0 and rect_h > 0:
                                proc_ann_resized = cv2.resize(proc_ann, (rect_w, rect_h), interpolation=cv2.INTER_NEAREST)
                                disp[disp_y1:disp_y2, disp_x1:disp_x2] = proc_ann_resized
                                cv2.rectangle(disp, (disp_x1, disp_y1), (disp_x2, disp_y2), (0, 255, 0), 1)
                        except Exception as e:
                            print(f"Err overlay/resize:{e}");
                else:
                    if proc_ann.shape[:2] != disp.shape[:2]:
                        disp = cv2.resize(proc_ann, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        disp = proc_ann

            elif crop:
                if packet.get('original_frame') is None: return
                orig_h, orig_w = packet['original_frame'].shape[:2]
                prev_h, prev_w = disp.shape[:2]
                if orig_w > 0 and orig_h > 0:
                    scale_w = prev_w / float(orig_w); scale_h = prev_h / float(orig_h)
                    disp_x1 = int(crop[0] * scale_w); disp_y1 = int(crop[1] * scale_h)
                    disp_x2 = int(crop[2] * scale_w); disp_y2 = int(crop[3] * scale_h)
                    cv2.rectangle(disp, (disp_x1, disp_y1), (disp_x2, disp_y2), (0, 255, 0), 1)

            self.latest_processed_frame = disp.copy()
            y=20; overlay_h = 160; overlay=disp.copy(); cv2.rectangle(overlay,(5,5),(200,overlay_h),(0,0,0),-1)
            disp = cv2.addWeighted(overlay,0.6,disp,0.4,0)
            def draw(l, v, c=(0,255,0)): nonlocal y; txt=f"{l:<9}: {v:>5.1f}"; cv2.putText(disp,txt,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,c,1,cv2.LINE_AA); y+=18
            draw("LATENCY(ms)",0,(255,255,0)); y-=18; draw("PreProc",lat['PreProc']); draw("Inf Q Wt",lat['InfQWait']); draw("Inference",lat['Infer']);
            draw("GUI Q Wt",lat['GUIQWait']); draw("GUI Updt",lat['GUIUpdate']); draw("Cap->CSV",lat['CapToCSV']); draw("E2E Disp",lat['E2E'],(255,255,0))

            if self.is_recording:
                self.recording_buffer.append(disp)

            self.orig_lbl.setPixmap(self.to_pixmap(preview_frame))
            self.proc_lbl.setPixmap(self.to_pixmap(disp))

        elif packet_type == 'preview':
            self.orig_lbl.setPixmap(self.to_pixmap(preview_frame))
            if self.is_recording:
                self.recording_buffer.append(preview_frame)

    def to_pixmap(self, frame):
        try:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); h,w,ch=rgb.shape
            if h<=0 or w<=0: return QPixmap()
            qimg=QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888); pix=QPixmap.fromImage(qimg.copy())
            target=self.proc_lbl or self.orig_lbl
            if target and not pix.isNull(): return pix.scaled(target.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
            else: return pix
        except Exception as e: print(f"Err pixmap:{e}"); return QPixmap()

    def closeEvent(self, event):
        print("Close event. Stopping all threads...");
        self.main_shutdown_event.set()
        
        if self.inference_manager_thread and self.inference_manager_thread.is_alive():
            print("Joining Inference Manager...")
            self.inference_shutdown_event.set()
            self.inference_manager_thread.join(timeout=7)
            
        if self.video_processing_thread and self.video_processing_thread.is_alive():
            print("Joining Video Thread...")
            self.video_processing_thread.join(timeout=3)

        if self.gui_update_worker and self.gui_update_worker.isRunning():
            print("Wait GUI...");
            self.gui_update_worker.wait(1000);

        if self.is_recording:
            self.is_recording = False
            self.recording_buffer = []
            print("Recording stopped on close.")
        
        print("Shutdown complete. Accepting close event.")
        event.accept()


def main():
    # Check for DLC availability
    try:
        from deeplabcut.pose_estimation_pytorch import get_pose_inference_runner
    except ImportError:
        app=QApplication(sys.argv)
        QMessageBox.critical(None,"Dependency Error","DeepLabCut could not be imported. Please ensure it is installed correctly.")
        sys.exit(1)

    if sys.platform!='win32':
        try:
            mp.set_start_method("spawn", force=True); print("Multiprocessing start method set to 'spawn'.")
        except RuntimeError:
            print("Warning: Could not force 'spawn' start method.")

    app=QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
