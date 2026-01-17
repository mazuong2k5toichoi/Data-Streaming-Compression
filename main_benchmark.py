







import cv2
import os
import time
import gc
import numpy as np
import av
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms
from dahuffman import HuffmanCodec
from fractions import Fraction
import psutil
# --- Try to import your specific model ---
try:
    from mcucoder.model import MCUCoder
except ImportError:
    print("WARNING: Could not import MCUCoder model. The MCUCoder benchmark will fail.")
    MCUCoder = None

# Import shared tools (Monitor, calculate_psnr, calculate_bpp)
# Assuming these are available in benchmark_utils.py or pasted above in your environment
from benchmark_utils import Monitor, calculate_psnr, calculate_bpp, calculate_msssim

# --- 1. ABSTRACT BASE FOR ALGORITHMS ---
class BaseAlgorithm:
    def __init__(self, width, height, **kwargs):
        self.width = width
        self.height = height
        self.kwargs = kwargs

    def get_name(self):
        raise NotImplementedError

    def encode_frame(self, frame):
        raise NotImplementedError

    def decode_frame(self, encoded_data):
        raise NotImplementedError

    def close(self):
        pass

# --- 2. HELPER FUNCTIONS ---
def _quantization_and_huffman(data, filter_number, codec, step):
    """
    Quantize data and encode it using Huffman.
    """
    min_val = codec['min'][filter_number]
    max_val = codec['max'][filter_number]
    
    data = data.detach()
    quantized_data = ((data - min_val) / (max_val - min_val) * 255).to(dtype=torch.uint8)
    
    # Apply step for quantization
    quantized_data = (quantized_data // step).to(dtype=torch.uint8)
    
    # Huffman Encoding
    quantized_data_np = quantized_data.cpu().numpy().flatten()
    huffman_codec = codec['codec'][filter_number]
    encoded = huffman_codec.encode(quantized_data_np)
    
    return len(encoded) * 8 # Return bits

# --- 3. CONCRETE ALGORITHMS ---

class MJPEGAlgorithm(BaseAlgorithm):
    def __init__(self, width, height, **kwargs):
        super().__init__(width, height, **kwargs)
        self.quality = kwargs.get('quality', 90)

    def get_name(self):
        return f"MJPEG_Q{self.quality}"

    def encode_frame(self, frame):
        _ , jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return jpeg_bytes, len(jpeg_bytes) * 8

    def decode_frame(self, encoded_data):
        return cv2.imdecode(encoded_data, cv2.IMREAD_COLOR)

class H264PyAVAlgorithm(BaseAlgorithm):
    def __init__(self, width, height, **kwargs):
        super().__init__(width, height, **kwargs)
        self.crf = kwargs.get('crf', 23)
        self.output_path = f"temp_h264_crf{self.crf}.mp4"
        fps_float = kwargs.get('fps', 30)
        self.framerate = Fraction(fps_float).limit_denominator()
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def get_name(self):
        return f"H264_CRF{self.crf}"

    def batch_encode(self, frames):
        output_container = av.open(self.output_path, mode='w')
        stream = output_container.add_stream('h264', rate=self.framerate)
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': str(self.crf)}
        
        encode_times = []
        for frame in tqdm(frames, desc=f"Encoding H.264 (CRF {self.crf})", leave=False):
            av_frame = av.VideoFrame.from_ndarray(frame[:, :, ::-1], format='rgb24')
            t0 = time.perf_counter()
            packets = stream.encode(av_frame)
            t1 = time.perf_counter()
            encode_times.append((t1 - t0) * 1000)
            for packet in packets:
                output_container.mux(packet)
        
        for packet in stream.encode(None):
            output_container.mux(packet)
            
        output_container.close()
        total_bits = os.path.getsize(self.output_path) * 8
        return encode_times, total_bits

    def batch_decode(self):
        input_container = av.open(self.output_path)
        decode_times = []
        decoded_frames = []
        for frame in tqdm(input_container.decode(video=0), desc="Decoding H.264", leave=False):
            t0 = time.perf_counter(); _ = frame.to_ndarray(); t1 = time.perf_counter()
            decode_times.append((t1 - t0) * 1000)
            decoded_frames.append(frame.to_ndarray(format='bgr24'))
        input_container.close()
        return decoded_frames, decode_times
        
    def close(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
class MCUCoderAlgorithm(BaseAlgorithm):
    def __init__(self, width, height, **kwargs):
        super().__init__(width, height, **kwargs)
        # FIX: Force CPU explicitly
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        # torch.set_num_threads(1)
        # --- BENCHMARKING BY FILTER COUNT ---
        self.channels = self.kwargs.get('channels', 12) 
        self.quant_step = kwargs.get('quant_step', 2)

        # Model Loading
        self.model = MCUCoder().to(self.device)
        model_path = self.kwargs.get('model_path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"MCUCoder model not found at: {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            # transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
        ])

        # Codec Creation
        original_frames = self.kwargs.get('original_frames')
        if not original_frames:
            raise ValueError("MCUCoder requires 'original_frames'.")
        
        print(f"   [MCUCoder] Creating Huffman Codec (Step={self.quant_step}, Filters={self.channels})...")
        self.codec = self._create_codec(original_frames)
        
        # --- OPTIMIZATION: PRE-CALCULATE TENSORS ---
        # We prepare these once to avoid creating them every frame
        self._prepare_quantization_params()

    def get_name(self):
        return f"MCUCoder_F{self.channels:02d}"

    def _create_codec(self, frames):
        # (Same as your original implementation)
        codec_setting = {'min': {}, 'max': {}, 'codec': {}}
        sample_frames = frames[::5] if len(frames) > 50 else frames

        batch_tensors = []
        for frame in tqdm(sample_frames, desc="Init Codec", leave=False):
            img = Image.fromarray(frame[:, :, ::-1]) 
            batch_tensors.append(self.transform(img))
        
        images = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            encoded = self.model.encoder(images)

        for i in range(self.channels):
            data = encoded[:, i, :, :].reshape(-1).cpu().numpy()
            min_, max_ = np.min(data), np.max(data)
            
            quantized_data = ((data - min_) / (max_ - min_) * 255).astype(np.uint8)
            quantized_data = (quantized_data // self.quant_step).astype(np.uint8)
            
            max_symbol = 255 // self.quant_step
            symbols = np.arange(0, max_symbol + 1)
            full_data = np.concatenate([quantized_data, symbols])

            codec = HuffmanCodec.from_data(np.unique(full_data))
            codec_setting['min'][i], codec_setting['max'][i], codec_setting['codec'][i] = min_, max_, codec

        return codec_setting

    def _prepare_quantization_params(self):
        """
        Pre-allocates Min and Range tensors on the GPU/Device.
        Shape: (1, C, 1, 1) to allow broadcasting against (1, C, H, W).
        """
        mins = [self.codec['min'][i] for i in range(self.channels)]
        maxs = [self.codec['max'][i] for i in range(self.channels)]
        
        # Create vectors
        min_t = torch.tensor(mins, device=self.device, dtype=torch.float32)
        max_t = torch.tensor(maxs, device=self.device, dtype=torch.float32)
        
        # Reshape for broadcasting: (1, Channels, 1, 1)
        self.min_tensor = min_t.view(1, -1, 1, 1)
        self.range_tensor = (max_t - min_t).view(1, -1, 1, 1)
        
        # Avoid division by zero edge case
        self.range_tensor[self.range_tensor == 0] = 1e-6

    def encode_frame(self, frame):
        img = Image.fromarray(frame[:, :, ::-1])
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        timings = {}

        with torch.no_grad():
            # 1. NN Encoder
            t_nn_start = time.perf_counter()
            encoded_latent = self.model.encoder(tensor)
            
            # Zero out unused channels (if channels < model output)
            # Slice to only keep the channels we are analyzing
            encoded_active = encoded_latent[:, :self.channels, :, :]
            
            t_nn_end = time.perf_counter()
            timings['nn_ms'] = (t_nn_end - t_nn_start) * 1000

            # 2. Vectorized Quantization & Huffman
            encode_monitor = Monitor()
            encode_monitor.start(print_baseline=False)
            t_quant_start = time.perf_counter()
            
            # --- STEP A: VECTORIZED QUANTIZATION (GPU) ---
            # (X - Min) / Range * 255
            # We do this for ALL channels in one parallel operation
            normalized = (encoded_active - self.min_tensor) / self.range_tensor
            quantized = (normalized * 255).clamp(0, 255).to(torch.uint8)
            
            # Apply Step
            quantized = (quantized // self.quant_step)
            
            # --- STEP B: BULK TRANSFER (GPU -> CPU) ---
            # Move once. Flatten spatial dims.
            # Shape becomes (Channels, H*W)
            quantized_np = quantized.cpu().numpy().reshape(self.channels, -1)
            
            # --- STEP C: HUFFMAN ENCODING (CPU) ---
            # We still loop here because Huffman tables are unique per channel,
            # but the math and transfer overhead is gone.
            total_bits = 0
            for i in range(self.channels):
                # encode() expects a flat array. We row-slice the numpy array.
                encoded_bytes = self.codec['codec'][i].encode(quantized_np[i])
                total_bits += len(encoded_bytes) * 8
            
            t_quant_end = time.perf_counter()
            peak_encode_ram = encode_monitor.stop() 
            timings['quant_ms'] = (t_quant_end - t_quant_start) * 1000
        
        # Reconstruct the full tensor for return (fill zeros for dropped channels)
        # (Only needed if the benchmarking caller expects the full 12-channel shape)
        if self.channels < 12:
            full_latent = torch.zeros_like(encoded_latent)
            full_latent[:, :self.channels, :, :] = encoded_active
            encoded_active = full_latent

        return encoded_active, total_bits, timings, peak_encode_ram
    # def encode_frame(self, frame):
    #     # 1. Prepare Data
    #     img = Image.fromarray(frame[:, :, ::-1])
    #     tensor = self.transform(img).unsqueeze(0).to(self.device)

    #     # --- CHANGE 1: INPUT RAM ---
    #     # The MCU reads raw RGB bytes (uint8), so we count 1 byte per pixel.
    #     # 224 * 224 * 3 * 1 byte = ~150 KB
    #     ram_input_kb = (tensor.nelement() * 1) / 1024.0  # Force 1 byte (UINT8)

    #     timings = {}

    #     with torch.no_grad():
    #         t_nn_start = time.perf_counter()
    #         encoded_latent = self.model.encoder(tensor)
    #         encoded_active = encoded_latent[:, :self.channels, :, :]
    #         t_nn_end = time.perf_counter()
    #         timings['nn_ms'] = (t_nn_end - t_nn_start) * 1000

    #         # --- CHANGE 2: LATENT RAM ---
    #         # The paper uses "INT8 quantized encoder"[cite: 63, 86].
    #         # This means activations and outputs are stored as 1-byte integers.
    #         # We simulate this by multiplying by 1 instead of 4.
    #         ram_latent_kb = (encoded_active.nelement() * 1) / 1024.0

    #         # --- Quantization & Huffman (Standard) ---
    #         t_quant_start = time.perf_counter()
            
    #         # Note: We don't track 'normalized' tensor RAM here because in an optimized 
    #         # INT8 MCU pipeline, normalization is fused into the quantization step.
            
    #         normalized = (encoded_active - self.min_tensor) / self.range_tensor
    #         quantized = (normalized * 255).clamp(0, 255).to(torch.uint8)
            
    #         # Output bitstream buffer (tracked as part of usage)
    #         ram_quant_kb = (quantized.nelement() * 1) / 1024.0

    #         quantized = (quantized // self.quant_step)
    #         quantized_np = quantized.cpu().numpy().reshape(self.channels, -1)
            
    #         total_bits = 0
    #         for i in range(self.channels):
    #             encoded_bytes = self.codec['codec'][i].encode(quantized_np[i])
    #             total_bits += len(encoded_bytes) * 8
            
    #         t_quant_end = time.perf_counter()
    #         timings['quant_ms'] = (t_quant_end - t_quant_start) * 1000
        
    #     if self.channels < 12:
    #         full_latent = torch.zeros_like(encoded_latent)
    #         full_latent[:, :self.channels, :, :] = encoded_active
    #         encoded_active = full_latent

    #     # --- CHANGE 3: PEAK RAM CALCULATION ---
    #     # On an MCU (TFLite Micro), memory is reused (Arena allocator).
    #     # The Peak is typically: Input Buffer + Largest Activation Buffer.
    #     # Since we don't know the exact internal activation size without TFLite, 
    #     # we estimate Peak as Input + Output (Latent), which must coexist.
        
    #     peak_enc_ram = ram_input_kb + ram_latent_kb

    #     return encoded_active, total_bits, timings, peak_enc_ram
    
    def decode_frame(self, encoded_data):
        with torch.no_grad():
            decoded_tensor = self.model.decoder(encoded_data).squeeze(0)
        return decoded_tensor

    def close(self):
        del self.model
        del self.codec
        if hasattr(self, 'min_tensor'): del self.min_tensor
        if hasattr(self, 'range_tensor'): del self.range_tensor
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

# --- 4. BENCHMARK RUNNER ---

def run_benchmark(algorithm: BaseAlgorithm, original_frames: list, fps: float):
    num_frames = len(original_frames)
    height, width, _ = original_frames[0].shape
    pixels_per_frame = width * height

    # --- VIDEO WRITER SETUP ---
    output_video_path = f"/mnt/c/Users/User/Desktop/bku/251/MCUCoder/MCUCoder/testt/jocky/decoded_{algorithm.get_name()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Logs: [enc_ms, dec_ms, psnr, mssim, bpp, ram_mb]
    logs = np.zeros((num_frames, 6), dtype=np.float32) 
    mcucoder_logs = None
    
    if isinstance(algorithm, MCUCoderAlgorithm):
        mcucoder_logs = np.zeros((num_frames, 2), dtype=np.float32) # [nn_ms, quant_ms]

    gc.collect()
    
    # --- Batch Mode (H.264) ---
    if isinstance(algorithm, H264PyAVAlgorithm):
        encode_monitor = Monitor()
        encode_monitor.start(print_baseline=False)
        enc_times, total_bits = algorithm.batch_encode(original_frames)
        peak_ram = encode_monitor.stop() # Peak during entire batch process
        
        dec_frames, dec_times = algorithm.batch_decode()
        
        for frame in dec_frames:
            out.write(frame)

        avg_bpp = total_bits / (num_frames * pixels_per_frame)
        for i in range(min(num_frames, len(dec_frames))):
            logs[i, 0] = enc_times[i]
            logs[i, 1] = dec_times[i]
            logs[i, 2] = calculate_psnr(original_frames[i], dec_frames[i])
            logs[i, 3] = calculate_msssim(original_frames[i], dec_frames[i])
            logs[i, 4] = avg_bpp
            logs[i, 5] = peak_ram # Same peak for all frames in batch

    # --- Frame Mode (MJPEG, MCUCoder) ---
    else:
        for i in tqdm(range(num_frames), desc=f"Run {algorithm.get_name()}", leave=False):
            original = original_frames[i]
            
            # Encode
            if isinstance(algorithm, MCUCoderAlgorithm):
                encoded, bits, timings, peak_ram = algorithm.encode_frame(original)
                logs[i, 0] = timings['nn_ms'] + timings['quant_ms']
                logs[i, 5] = peak_ram
                mcucoder_logs[i, 0] = timings['nn_ms']
                mcucoder_logs[i, 1] = timings['quant_ms']
            else:
                monitor = Monitor()
                monitor.start(print_baseline=False)
                t0 = time.perf_counter()
                encoded, bits = algorithm.encode_frame(original)
                t1 = time.perf_counter()
                peak_ram = monitor.stop()
                logs[i, 0] = (t1 - t0) * 1000
                logs[i, 5] = peak_ram
            
            # Decode
            t0 = time.perf_counter()
            decoded = algorithm.decode_frame(encoded)
            t1 = time.perf_counter()
            logs[i, 1] = (t1 - t0) * 1000

            # --- VIDEO WRITING ---
            if isinstance(decoded, torch.Tensor):
                # Convert tensor (C, H, W) [0,1] RGB to numpy (H, W, C) [0,255] BGR
                decoded_np = decoded.permute(1, 2, 0).cpu().detach().numpy() * 255.0
                decoded_bgr = cv2.cvtColor(decoded_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(decoded_bgr)
            else:
                # MJPEG output is already a BGR numpy array
                out.write(decoded)

            # Metrics
            logs[i, 2] = calculate_psnr(original, decoded)
            logs[i, 3] = calculate_msssim(original, decoded)
            logs[i, 4] = calculate_bpp(bits, width, height)

    out.release()
    algorithm.close()

    # --- SAVE LOGS ---
    log_filename = f"results_{algorithm.get_name().lower()}.csv"
    header = "enc_time_ms,dec_time_ms,psnr,mssim,bpp,enc_ram_kb"
    if mcucoder_logs is not None:
        combined_logs = np.hstack((logs, mcucoder_logs))
        header += ",nn_ms,quant_ms"
        np.savetxt(log_filename, combined_logs, delimiter=",", header=header, comments='')
    else:
        np.savetxt(log_filename, logs, delimiter=",", header=header, comments='')
    
    # --- SUMMARY ---
    summary = {
        'Algorithm': algorithm.get_name(),
        'Avg BPP': np.mean(logs[:, 4]),
        'Avg PSNR': np.mean(logs[:, 2]),
        'Avg MS-SSIM': np.mean(logs[:, 3]),
        'Avg Enc Time': np.mean(logs[:, 0]),
        # --- UPDATED: Peak RAM ---
        'Peak Enc RAM (KB)': np.max(logs[:, 5]), 
    }
    
    if isinstance(algorithm, MCUCoderAlgorithm):
        summary['NN Time'] = np.mean(mcucoder_logs[:, 0])
        summary['Quant Time'] = np.mean(mcucoder_logs[:, 1])
    else:
        summary['NN Time'] = 0
        summary['Quant Time'] = 0

    return summary

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    VIDEO_PATH = '/mnt/c/Users/User/Desktop/bku/251/MCUCoder/MCUCoder/video/jocky_resized.mp4'
    MODEL_PATH = 'MCUCoder1M300k196MSSSIM.pth'
    
    if not os.path.exists(VIDEO_PATH):
        print("Video not found.")
        exit()

    print("Loading video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    # for _ in range(20):
    #     ret, f = cap.read()
    #     if not ret:
    #         break
    #     frames.append(f)
    # cap.release()
    # print(f"Loaded {len(frames)} frames.")
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    final_results = []

    # MJPEG (Quality 10-90)
    mjpeg_configs = [1, 10, 30, 50, 70, 90]

    # H.264 (CRF 45-18)
    h264_configs = [45, 35, 28, 23, 18]

    # --- UPDATED: MCUCoder Filter Sweep ---
    # Benchmarking by Number of Filters (1, 3, 6, 9, 12)
    mcucoder_filters = [1, 3, 6, 9, 12]

    print("\n--- MJPEG Sweep ---")
    for q in mjpeg_configs:
        algo = MJPEGAlgorithm(width, height, quality=q)
        final_results.append(run_benchmark(algo, frames, fps))

    print("\n--- H.264 Sweep ---")
    for crf in h264_configs:
        algo = H264PyAVAlgorithm(width, height, fps=fps, crf=crf)
        final_results.append(run_benchmark(algo, frames, fps))

    print("\n--- MCUCoder Sweep (By Filters) ---")
    for num_filters in mcucoder_filters:
        algo = MCUCoderAlgorithm(
            width=width, height=height, 
            model_path=MODEL_PATH, 
            original_frames=frames,
            channels=num_filters,  # Using number of filters
            quant_step=4           # Constant quantization step
        )
        final_results.append(run_benchmark(algo, frames, fps))

    # --- DISPLAY & SAVE SUMMARY ---
    df = pd.DataFrame(final_results)
    
    # Updated column list to include Peak RAM
    cols = ['Algorithm', 'Avg BPP', 'Avg PSNR', 'Avg MS-SSIM', 'Avg Enc Time', 'NN Time', 'Quant Time', 'Peak Enc RAM (KB)']
    df = df[cols]
    
    print("\n" + "="*90)
    print("FINAL BENCHMARK SUMMARY (Sorted by BPP)")
    print("="*90)
    print(df.sort_values(by=['Algorithm', 'Avg BPP']).to_string(index=False))
    print("="*90)

    df.to_csv("benchmark_summary.csv", index=False)
    print("Overall summary saved to 'benchmark_summary.csv'")