# src/dlc_live_torch/workers.py

import os
import time
import cv2
import numpy as np
import threading
import multiprocessing as mp
from PyQt5.QtCore import QThread, pyqtSignal

# Conditional import for psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Local import from the package
from .inference import inference_worker

class InferenceProcessManager(threading.Thread):
    """ Manages inference process with hot-swap. """
    def __init__(self, settings, active_queue_ref, results_queue, shutdown_event, processed_counter, csv_counter):
        super().__init__(); self.settings = settings; self.active_queue_ref = active_queue_ref; self.results_queue = results_queue
        self.main_shutdown_event = shutdown_event
        self.daemon = True; self.processes = [None, None]
        self.process_queues = [mp.Queue(maxsize=1), mp.Queue(maxsize=1)]
        self.shutdown_events=[mp.Event(), mp.Event()]; self.ready_events=[mp.Event(), mp.Event()]; self.active_idx=0
        self.swap_headstart_seconds = 5.0; self.processed_counter = processed_counter; self.csv_counter = csv_counter

    def _start_process(self, idx):
        if self.processes[idx] and self.processes[idx].is_alive(): return
        print(f"Mgr: Start proc {idx}..."); self.shutdown_events[idx].clear(); self.ready_events[idx].clear()
        args = (self.settings, self.process_queues[idx], self.results_queue, self.shutdown_events[idx], self.ready_events[idx], self.processed_counter, self.csv_counter)
        self.processes[idx] = mp.Process(target=inference_worker, args=args)
        self.processes[idx].start()

    def _stop_process(self, idx):
        proc=self.processes[idx]; q=self.process_queues[idx]
        if proc and proc.is_alive():
            print(f"Mgr: Stop proc {idx}..."); self.shutdown_events[idx].set()
            while not q.empty():
                try: q.get_nowait()
                except Exception: break
            proc.join(timeout=3);
            if proc.is_alive():
                print(f"Mgr: Force term proc {idx}.")
                proc.terminate()
                proc.join(timeout=2)
            print(f"Mgr: Proc {idx} stopped.")
        self.processes[idx] = None;
        while not q.empty():
            try: q.get_nowait()
            except Exception: break

    def run(self):
        if not PSUTIL_AVAILABLE:
            print("!!! WARN: psutil N/A.");
            self._start_process(0); self.active_queue_ref.set(self.process_queues[0])
            self.main_shutdown_event.wait();
            self._stop_process(0)
            print("Mgr: Stopped."); return
            
        limit=self.settings['ram_threshold_gb']; trig=limit*0.9; self._start_process(self.active_idx); self.active_queue_ref.set(self.process_queues[self.active_idx]); standby=False
        
        while not self.main_shutdown_event.is_set():
            active=self.processes[self.active_idx]; mem=0
            if active and active.is_alive():
                try: mem = psutil.Process(active.pid).memory_info().rss/(1024**3)
                except psutil.NoSuchProcess:
                    time.sleep(1); continue
            else:
                if not self.main_shutdown_event.is_set():
                    print(f"Mgr: Proc {self.active_idx} died. Restart.")
                    self._start_process(self.active_idx)
                    self.active_queue_ref.set(self.process_queues[self.active_idx])
                    standby=False; time.sleep(5)
                continue
            
            if not standby and mem>trig:
                stby_idx=1-self.active_idx
                print(f"Mgr: Mem {mem:.2f}/{limit:.2f} GB. Pre-start {stby_idx}...")
                self._start_process(stby_idx); standby=True
            
            if mem>limit:
                print(f"Mgr: RAM breach ({mem:.2f}/{limit:.2f} GB). Swap."); stby_idx=1-self.active_idx
                if not self.processes[stby_idx] or not self.processes[stby_idx].is_alive():
                    print(f"Mgr: Standby {stby_idx} N/A. Start."); self._start_process(stby_idx); time.sleep(1)
                print(f"Mgr: Wait standby {stby_idx} ready..."); ready=self.ready_events[stby_idx].wait(timeout=self.swap_headstart_seconds)
                if ready: print(f"Mgr: SWAP {self.active_idx}->{stby_idx}.")
                else: print(f"Mgr: WARN - Standby {stby_idx} not ready.")
                self.active_queue_ref.set(self.process_queues[stby_idx]); old=self.active_idx; self.active_idx=stby_idx; standby=False; self._stop_process(old)
            
            time.sleep(1)
            
        print("Mgr: Shutdown signal received.")
        self._stop_process(0); self._stop_process(1);
        print("Mgr: Stopped.")

class VideoProcessingThread(threading.Thread):
    """ Reads frames, applies cropping/pre-processing, puts latest in queue. """
    def __init__(self, settings, active_queue_ref, preview_queue, shutdown_event, command_queue, captured_counter, enqueued_counter, crop_manager):
        super().__init__(); self.settings = settings; self.active_queue_ref = active_queue_ref
        self.preview_queue = preview_queue
        self.shutdown_event = shutdown_event; self.command_queue = command_queue; self.daemon = True
        self.cap = None; self.flat_img = None; self.captured_counter = captured_counter
        self.enqueued_counter = enqueued_counter; self.crop_manager = crop_manager; print("[VidThread] Init.")
        
        self.preview_width = 640
        self.preview_height = 360
        try:
            cam_w = int(self.settings.get('cam_width', 640))
            cam_h = int(self.settings.get('cam_height', 480))
            if cam_w > 0 and cam_h > 0:
                aspect_ratio = cam_w / float(cam_h)
                self.preview_height = int(self.preview_width / aspect_ratio)
            print(f"Preview size set to: {self.preview_width}x{self.preview_height}")
        except Exception as e:
            print(f"Warn: Could not auto-set preview size: {e}. Using {self.preview_width}x{self.preview_height}.")

        if self.settings['method'] == 'Flat-field':
            path = self.settings['flat_image_path']
            if path and os.path.exists(path):
                try: self.flat_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                except Exception as e: print(f"!!! WARN: Load flat fail '{path}': {e}")
            elif path: print(f"!!! WARN: Flat N/F: {path}")

    def process_frame(self, frame):
        method = self.settings.get('method', 'None')
        try:
            if method == 'Flat-field' and self.flat_img is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if gray.shape != self.flat_img.shape:
                    try: self.flat_img = cv2.resize(self.flat_img, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                    except Exception as e: print(f"WARN: Resize flat img failed: {e}"); return frame
                corr = cv2.divide(gray, self.flat_img.astype(np.float32) + 1e-6, scale=255.0)
                return cv2.cvtColor(np.uint8(np.clip(corr, 0, 255)), cv2.COLOR_GRAY2BGR)
            elif method == 'Morphological Opening':
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)); return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kern)
        except Exception as e: print(f"!!! WARN: Preproc '{method}' err: {e}")
        return frame

    def apply_camera_commands(self):
        try:
            while self.command_queue and not self.command_queue.empty():
                cmd = self.command_queue.get_nowait()
                if 'property' in cmd and 'value' in cmd and self.cap: self.cap.set(cmd['property'], cmd['value'])
        except Exception: pass

    def run(self):
        try:
            print("[VidThread] Run."); source = self.settings['camera_path']
            if self.settings['camera_source'] == 'USB Webcam': source = int(source)
            print(f"[VT] Open: {source}"); self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print(f"!!! CRIT: Open fail: {source}"); raise IOError("Cannot open video source")
            print("[VT] Source open."); buffer_ok = self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[VT] Set BUFFERS=1, ok={buffer_ok}")

            if self.settings['camera_source'] == 'USB Webcam':
                print("[VT] Set USB props..."); props = []
                props.append(self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings['cam_width']))
                props.append(self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings['cam_height']))
                props.append(self.cap.set(cv2.CAP_PROP_FPS, self.settings['cam_fps']))
                props.append(self.cap.set(cv2.CAP_PROP_EXPOSURE, self.settings['exposure']))
                props.append(self.cap.set(cv2.CAP_PROP_GAIN, self.settings['gain']))
                props.append(self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, self.settings['white_balance']))
                print(f"[VT] Req: {self.settings['cam_width']}x{self.settings['cam_height']} @ {self.settings['cam_fps']} FPS")
                print(f"[VT] Act: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {self.cap.get(cv2.CAP_PROP_FPS)} FPS")
                if not all(props): print("!!! WARN: Prop set fail.")
                else: print("[VT] USB props set.")
        except Exception as e:
            print(f"!!! CRIT INIT ERR: {e}"); self.shutdown_event.set();
            if self.cap:
                self.cap.release();
            return

        print("[VT] Loop start..."); frames=0; puts=0; inf_puts=0
        while not self.shutdown_event.is_set():
            self.apply_camera_commands(); ret, frame = self.cap.read(); t_cap = time.monotonic()

            if not ret:
                print(f"[VT] Read fail (ret={ret}). End?");
                if self.settings['camera_source']=='Video File':
                    print("!!! [VT] End of video file. Re-opening...")
                    self.cap.release()
                    source = self.settings['camera_path']
                    self.cap = cv2.VideoCapture(source)
                    if not self.cap.isOpened():
                        print(f"!!! [VT] Failed to re-open video file. Stopping thread.")
                        self.shutdown_event.set(); break
                    else:
                        print("[VT] Video file restarted.")
                        continue
                elif self.settings['camera_source']!='Video File' and self.cap.isOpened():
                    print("!!! Err read but cam open..."); time.sleep(0.1); continue
                else:
                    print("!!! Cam closed/file end."); self.shutdown_event.set(); break

            frames+=1; self.captured_counter.value+=1;
            
            try:
                preview_frame = cv2.resize(frame, (self.preview_width, self.preview_height), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print(f"!!! [VT] Resize fail: {e}, frame shape {frame.shape}")
                continue
            
            orig = frame.copy()
            
            crop = self.crop_manager.get()
            proc_frame_for_packet = frame
            if crop:
                x1,y1,x2,y2 = crop; h,w = frame.shape[:2]; x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
                if x1<x2 and y1<y2: proc_frame_for_packet = frame[y1:y2, x1:x2]
                else: print("Warn: Invalid crop"); self.crop_manager.clear(); crop=None

            proc = self.process_frame(proc_frame_for_packet)

            packet = {'original_frame': orig, 'preview_frame': preview_frame, 'processed_frame': proc,
                        'timestamps': {'capture': t_cap, 'processed': time.monotonic(), 'enqueued_for_inference': -1},
                        'crop_coords': crop}

            try:
                preview_packet = {'preview_frame': preview_frame, 'type': 'preview', 'timestamps': {'capture': t_cap}, 'original_frame': orig}
                if self.preview_queue.full():
                    try: self.preview_queue.get_nowait()
                    except Exception: pass
                self.preview_queue.put_nowait(preview_packet)
                puts += 1
            except Exception:
                pass
                
            q = self.active_queue_ref.get()
            if q:
                try:
                    if not q.empty():
                        try: q.get_nowait()
                        except Exception: pass
                    packet['timestamps']['enqueued_for_inference'] = time.monotonic()
                    q.put_nowait(packet)
                    self.enqueued_counter.value+=1; inf_puts+=1
                except Exception: pass

        print(f"[VT] Exit loop. Read:{frames}, PreviewPuts:{puts}, InferPuts:{inf_puts}");
        if self.cap: self.cap.release(); print("VideoThread: Stopped.")


class GuiUpdateWorker(QThread):
    """ Gets latest results for GUI update at target FPS. """
    new_frame_ready = pyqtSignal(dict)
    def __init__(self, results_queue, preview_queue, shutdown_event, target_fps):
        super().__init__();
        self.results_queue = results_queue; self.preview_queue = preview_queue
        self.shutdown_event = shutdown_event
        self.target_interval = 1.0/target_fps if target_fps > 0 else 0; self.min_sleep = 0.001

    def run(self):
        while not self.shutdown_event.is_set():
            start = time.monotonic(); packet = None
            
            try:
                qsize = self.results_queue.qsize()
                if qsize > 1:
                    for _ in range(qsize - 1):
                        try: self.results_queue.get_nowait()
                        except Exception: break
                packet = self.results_queue.get_nowait()
                packet['type'] = 'processed'
                if 'timestamps' not in packet: packet['timestamps'] = {}
                packet['timestamps']['dequeued_for_gui'] = time.monotonic()
            except Exception:
                pass

            if not packet:
                try:
                    qsize = self.preview_queue.qsize()
                    if qsize > 1:
                        for _ in range(qsize - 1):
                            try: self.preview_queue.get_nowait()
                            except Exception: break
                    packet = self.preview_queue.get_nowait()
                    if 'type' not in packet: packet['type'] = 'preview'
                    if 'timestamps' not in packet: packet['timestamps'] = {}
                    packet['timestamps']['dequeued_for_gui'] = time.monotonic()
                except Exception:
                    pass

            if packet:
                self.new_frame_ready.emit(packet)

            elapsed = time.monotonic() - start; sleep_needed = self.target_interval - elapsed
            sleep = max(self.min_sleep, sleep_needed if sleep_needed > 0 else self.min_sleep); time.sleep(sleep)
        print("GuiUpdateWorker: Stopped.")
