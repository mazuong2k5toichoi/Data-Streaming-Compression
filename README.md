# Integration_Project-_Video_Data_Compression
My Integration project of learning about data compression, especially for video data compression on resource constrained devices.

You can install my slides and the final report via this link : [Slides](https://drive.google.com/drive/folders/1b1AOiBGUpbjDHMJsw_otm5IHJXBlL5Na?usp=sharing)

The original repositories of [MCUCoder](https://github.com/ds-kiel/MCUCoder)

## Specs

This repository contains learning modules and prototype work for video compression and IoT video streaming. The sections below describe each component, the learning goals, expected inputs/outputs, and minimal implementation notes.

### Compression technique / Compression technique p2
- Purpose: Learn and demonstrate core differences between lossless and lossy compression.
- Learning goals: understand entropy coding (Huffman/Arithmetic), run-length encoding, transform coding (DCT), quantization and their trade-offs (quality vs. rate).
- Inputs: raw or minimally encoded image frames (PNG, BMP, or YUV frames).
- Outputs: compressed bitstreams or files (examples: .bin, .crc'd payloads) and reconstructed frames for visual comparison.
- Success criteria: produce a lossless coder that exactly reconstructs input and a lossy coder that achieves measurable rate-quality trade-offs (PSNR/SSIM vs. bitrate).
- Notes: keep algorithms modular so you can swap entropy coder, transform, and quantizer. Document where each experiment lives (e.g., a folder per technique).

### H.264
- Purpose: Learn about H.264 (AVC) video compression and its practical use for efficient streaming/storage.
- Learning goals: understand I-frames/P-frames/B-frames, motion estimation, macroblocks/CTUs, intra/inter prediction, and baseline encoder pipeline.
- Inputs: raw video or sequence of frames (YUV recommended for reference testing).
- Outputs: H.264-compliant bitstreams (.mp4/.mkv containing H.264) and decoder-verified reconstructions.
- Success criteria: encode sample clips with a reference library or wrapper (libx264) and verify playback and bitrate/quality metrics.
- Notes: for resource constrained devices, study encoder presets and tune motion search/quantization to reduce CPU while keeping acceptable quality.

### IOVT compressions (IoT Video) and MCUCoder
- Purpose: Explore compression and streaming strategies tailored to IoT devices (microcontrollers, low-power cameras) and provide a small encoder ("MCUCoder") for constrained hardware.
- Learning goals: end-to-end streaming flow (capture → encode → packetize → transport), packet loss resilience, low-latency trade-offs, and power/CPU/memory profiling.
- Inputs: low-resolution frames (e.g., 320x240, grayscale/RGB565), short clips or live camera feeds from MCU-attached sensors.
- Outputs: small, packet-friendly compressed chunks for transport over constrained networks (MQTT/UDP/CoAP over lossy links), with metadata for reassembly and simple error detection.
- Success criteria: produce a working MCU-oriented encoder proof-of-concept (MCUCoder) that can run on a microcontroller simulator or low-end board, and stream to a host receiver that reconstructs frames with reasonable latency.
- Notes: prioritize simple, fast transforms and lightweight entropy coding (e.g., predictive coding + lightweight Huffman or LZ variants). Include a basic packet format and minimal error detection (CRC8/16).

## Contract (brief)
- Inputs: frame sequences (common formats: PNG, BMP, raw YUV, small camera output formats). Acceptable sizes and color formats should be documented per module.
- Outputs: compressed bitstreams and decoder-side reconstructed frames; metrics (bitrate, PSNR/SSIM), and example player-ready files for H.264 experiments.
- Error modes: unsupported formats, out-of-memory on MCU targets, packet loss in streaming; each module should log clear errors and fail gracefully.

## Edge cases and constraints
- Empty input or single-frame inputs — lossless coder must handle trivially.
- Extremely low bitrates — verify behavior of lossy modes and avoid catastrophic quality collapse.
- Packet loss and reordering for IoT streams — specify simple resync or keyframe intervals.
- Memory/CPU caps for MCU targets — document expected RAM/flash usage and provide a minimal configuration for target boards.


## Notes
- This spec is intentionally lightweight and educational. If you want, I can:
	- scaffold the module folders and example runner scripts,
	- add a small PSNR/SSIM measurement script and sample test vectors,
	- or produce a minimal MCUCoder prototype in C (or a simulator-friendly Python version).

---
_Updated: added high-level specs for the repository components (Compression technique, Compression technique p2, H.264, IOVT compressions, MCUCoder)._ 