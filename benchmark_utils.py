# In benchmark_utils.py

import resource
import os
import threading
import time
import numpy as np
import math
import torch
from fractions import Fraction
import cv2

class Monitor:
    """
    Monitors peak RSS memory growth during a code block.
    """
    def __init__(self):
        self.flag = False
        self.page_size = resource.getpagesize()
        self.thread = None
        self.peak_rss_kb = 0
        self.baseline_kb = 0
    def get_current_rss_kb(self):
        try:
            with open(f'/proc/{os.getpid()}/stat', 'r') as pstat:
                content = pstat.read().split()
                rss_pages = int(content[23])
                return (rss_pages * self.page_size) / 1024
        except Exception:
            return 0.0
    def _monitor(self):
        while self.flag:
            current_kb = self.get_current_rss_kb()
            growth = current_kb - self.baseline_kb
            if growth > self.peak_rss_kb:
                self.peak_rss_kb = growth
            time.sleep(0.0001)
    def start(self, print_baseline=True):
        self.baseline_kb = self.get_current_rss_kb()
        if print_baseline:
            print(f"   [Monitor] Baseline RAM: {self.baseline_kb / 1024:.2f} MB")
        self.flag = True
        self.peak_rss_kb = 0
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    def stop(self):
        self.flag = False
        if self.thread:
            self.thread.join(timeout=1)
        return self.peak_rss_kb


def calculate_psnr(original, compressed):
    """
    Calculates PSNR. Handles both NumPy arrays and PyTorch tensors.
    Assumes inputs are in range [0, 255] or [0, 1].
    """
    # --- START OF FIX ---
    # Handle PyTorch tensor output from the model
    if isinstance(compressed, torch.Tensor):
        # Model output is a (C, H, W) tensor in [0, 1] range.
        # We need to convert it to a (H, W, C) NumPy array in [0, 255] range.
        
        # 1. Permute dimensions from (C, H, W) to (H, W, C)
        compressed_tensor = compressed.permute(1, 2, 0)
        # 2. Convert to NumPy array
        compressed_np = compressed_tensor.cpu().detach().numpy()
        # 3. Scale from [0, 1] to [0, 255] and update the variable
        compressed = compressed_np * 255.0

    original = np.clip(original, 0, 255)
    compressed = np.clip(compressed, 0, 255)

    mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
    
    if mse == 0:
        return 100.0  # Or float('inf') for perfect match

    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def calculate_bpp(bits, width, height):
    """Calculates bits per pixel (BPP)."""
    pixels = width * height
    if pixels == 0:
        return 0.0
    return bits / pixels

def calculate_msssim(original, compressed):
    """
    Calculates MS-SSIM. Handles both NumPy arrays and PyTorch tensors.
    """
    # --- START OF CONVERSION ---
    # Handle PyTorch tensor output from the model for 'compressed'
    if isinstance(compressed, torch.Tensor):
        # Model output is a (C, H, W) tensor in [0, 1] range.
        # We need to convert it to a (H, W, C) NumPy array in [0, 255] range for resizing.
        compressed_np = compressed.permute(1, 2, 0).cpu().detach().numpy() * 255.0
    else:
        compressed_np = compressed

    # At this point, both `original` and `compressed_np` are NumPy arrays
    # in (H, W, C) format and [0, 255] range.

    # Convert to tensors for MS-SSIM calculation
    original_t = torch.from_numpy(original.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    compressed_t = torch.from_numpy(compressed_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Calculate MS-SSIM
    # data_range is 1.0 because the tensors are in [0, 1] range.
    # We use the PyTorch-MSSIM library which expects (N, C, H, W) tensors.
    from pytorch_msssim import ms_ssim
    msssim_val = ms_ssim(original_t, compressed_t, data_range=1.0, size_average=True)
    
    return msssim_val.item()
