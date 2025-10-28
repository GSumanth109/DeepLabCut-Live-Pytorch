# src/dlc_live_torch/config.py

import os
import yaml

CONFIG_FILE = "dlclive_config.yaml"

def load_app_config():
    """ Loads configuration from YAML file. """
    defaults = {
        'dlc_config_path': '', 'snapshot_path': '', 'pytorch_config_path': '',
        'camera_source': 'USB Webcam', 'camera_path': '0', 'cam_width': 640, 'cam_height': 480, 'cam_fps': 60,
        'exposure': -11, 'gain': 0, 'white_balance': 4000,
        'preproc_method': 'None', 'flat_image_path': '',
        'use_fp16': True, 'display_fps': 60, 'ram_threshold_gb': 16,
        'skeleton_confidence': 0.10, 'point_confidence': 0.60, 'show_skeleton': True,
        'save_csv': False, 'csv_output_path': '',
        'crop_coords': None
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = yaml.safe_load(f)
            if config:
                defaults.update(config)
            print(f"Loaded configuration from {CONFIG_FILE}")
            return defaults
        except Exception as e:
            print(f"Error loading {CONFIG_FILE}, using defaults: {e}")
            return defaults
    else:
        print(f"Config file {CONFIG_FILE} not found, using defaults.")
        return defaults

def save_app_config(config_dict):
    """ Saves configuration to YAML file. """
    global CONFIG_FILE
    try:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Error saving {CONFIG_FILE}: {e}")
