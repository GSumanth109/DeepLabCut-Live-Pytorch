# src/dlc_live_torch/inference.py

import os
import time
import cv2
import numpy as np
import torch
import gc
import csv
from deeplabcut.pose_estimation_pytorch import get_pose_inference_runner
from deeplabcut.core.config import read_config_as_dict

def inference_worker(settings, process_queue, results_queue, shutdown_event, ready_event, processed_counter, csv_counter):
    pose_runner = None; csv_writer = None; csv_file = None; frame_idx = 0; worker_pid = os.getpid()
    try:
        print(f"[Proc-{worker_pid}] Init model...");
        if not settings['pytorch_config_path'] or not os.path.exists(settings['pytorch_config_path']):
            print(f"!!! [Proc-{worker_pid}] ERROR: pytorch_config.yaml path missing or invalid: {settings['pytorch_config_path']}")
            ready_event.set(); return
        if not settings['config_path'] or not os.path.exists(settings['config_path']):
            print(f"!!! [Proc-{worker_pid}] ERROR: config.yaml path missing or invalid: {settings['config_path']}")
            ready_event.set(); return
            
        model_cfg = read_config_as_dict(settings['pytorch_config_path'])
        if 'method' not in model_cfg:
            main_cfg=read_config_as_dict(settings['config_path'])
            model_cfg['method'] = 'BottomUp' if not main_cfg.get('multianimalproject') else 'TopDown'
        
        pose_runner = get_pose_inference_runner( model_config=model_cfg, snapshot_path=settings['snapshot_path'], device="cuda" if torch.cuda.is_available() else "cpu" )
        if settings['use_fp16']: pose_runner.model.half()
        
        if settings['save_csv']:
            print(f"[Proc-{worker_pid}] Init CSV: {settings['csv_output_path']}")
            header = ['timestamp', 'frame_index', 'capture_to_csv_ms', 'inference_ms']; bodyparts = read_config_as_dict(settings['config_path'])['bodyparts']
            for bp in bodyparts: header.extend([f'{bp}_x', f'{bp}_y', f'{bp}_likelihood'])
            csv_file = open(settings['csv_output_path'], 'w', newline=''); csv_writer = csv.writer(csv_file); csv_writer.writerow(header)
        
        print(f"[Proc-{worker_pid}] Model ready."); ready_event.set()
    except Exception as e:
        print(f"!!! [Proc-{worker_pid}] INIT ERROR: {e}")
        import traceback
        traceback.print_exc()
        if csv_file:
            csv_file.close()
        ready_event.set(); return

    while not shutdown_event.is_set():
        try:
            packet = None
            try:
                packet = process_queue.get(timeout=0.1); packet['timestamps']['dequeued_for_inference'] = time.monotonic()
            except Exception: # queue.Empty
                if shutdown_event.is_set():
                    break
                continue

            rgb_frame = cv2.cvtColor(packet['processed_frame'], cv2.COLOR_BGR2RGB)
            inf_start = time.monotonic(); predictions = pose_runner.inference(images=[rgb_frame]); inf_end = time.monotonic()
            inference_ms = (inf_end - inf_start) * 1000

            packet['predictions'] = predictions[0]; packet['inference_time_ms'] = inference_ms; packet['timestamps']['inferred'] = inf_end
            csv_write_t = time.monotonic(); capture_to_csv_ms = (csv_write_t - packet['timestamps']['capture']) * 1000
            packet['capture_to_csv_ms'] = capture_to_csv_ms

            if csv_writer:
                row = [packet['timestamps']['capture'], frame_idx, f"{capture_to_csv_ms:.2f}", f"{inference_ms:.2f}"]
                keypoints = predictions[0]["bodyparts"]; crop = packet.get('crop_coords')
                
                if keypoints is not None and keypoints.size > 0:
                    if keypoints.ndim == 3: # Multi
                        for kp_set in keypoints:
                            for kp in kp_set: row.extend([kp[0]+(crop[0] if crop else 0), kp[1]+(crop[1] if crop else 0), kp[2]]) if isinstance(kp,(np.ndarray,list,tuple)) and len(kp)==3 else (row.extend(kp) if isinstance(kp,(np.ndarray,list,tuple)) and len(kp)==3 else row.extend(['NaN']*3))
                    elif keypoints.ndim == 2 and keypoints.shape[1] == 3: # Single
                            for kp in keypoints: row.extend([kp[0]+(crop[0] if crop else 0), kp[1]+(crop[1] if crop else 0), kp[2]]) if isinstance(kp,(np.ndarray,list,tuple)) and len(kp)==3 else (row.extend(kp) if isinstance(kp,(np.ndarray,list,tuple)) and len(kp)==3 else row.extend(['NaN']*3))
                    else: print(f"[P-{worker_pid}] Warn: Bad shape: {keypoints.shape}")
                else: num_bp = len(read_config_as_dict(settings['config_path'])['bodyparts']); row.extend(['NaN']*(num_bp*3))
                csv_writer.writerow(row); csv_counter.value += 1

            packet['timestamps']['enqueued_for_gui'] = time.monotonic()
            results_queue.put(packet); frame_idx += 1; processed_counter.value += 1
        except Exception as e:
            if not shutdown_event.is_set():
                import traceback
                print(f"!!! [Proc-{worker_pid}] CRASH loop: {e}")
                traceback.print_exc()
            break
            
    print(f"[Proc-{worker_pid}] Clean up..."); del pose_runner;
    if csv_file:
        csv_file.close()
    gc.collect();
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[Proc-{worker_pid}] Stopped.")
