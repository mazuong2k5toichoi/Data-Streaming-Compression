# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from mcucoder.model import MCUCoder
from tqdm import tqdm
from dahuffman import HuffmanCodec
import torch.nn.functional as F
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import argparse
import random
# import tensorflow as tf
import seaborn as sns
import logging

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

sns.set(font_scale=1.2)
plt.rc('legend', fontsize=10)
sns.set_style("whitegrid")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("video_enc_dec.log")  # Log to a file
    ]
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save output images and video")
    args = parser.parse_args()

    # Log parsed arguments
    logging.info(f"Parsed arguments: {args}")
    return args

# def encode_TFLite(model_path, X):
#     x_data = np.copy(X.to('cpu').numpy()) # the function quantizes the input, so we must make a copy
#     # Initialize the TFLite interpreter
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()[0]
#     output_details = interpreter.get_output_details()[0]
#     # Inputs will be quantized
#     input_scale, input_zero_point = input_details["quantization"]
#     if (input_scale, input_zero_point) != (0.0, 0):
#         x_data = x_data / input_scale + input_zero_point
#         x_data = x_data.astype(input_details["dtype"])
#     # Invoke the interpreter
#     predictions = np.empty((x_data.shape[0],12,28,28), dtype=output_details["dtype"])
#     for i in range(len(x_data)):
#         interpreter.set_tensor(input_details["index"], [x_data[i]])
#         interpreter.invoke()
#         predictions[i] = np.copy(interpreter.get_tensor(output_details["index"])[0])
#     # Dequantize output
#     output_scale, output_zero_point = output_details["quantization"]
#     if (output_scale, output_zero_point) != (0.0, 0):
#         predictions = predictions.astype(np.float32)
#         predictions = (predictions - output_zero_point) * output_scale
#     # todo reshape output into array for each exit
#     return torch.from_numpy(predictions).to('cuda')



class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        logging.info(f"Initializing VideoDataset with video path: {video_path}")

        # Get the total number of frames in the video
        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        logging.info(f"Loaded {self.total_frames} frames from video.")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, frame_index):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IndexError(f"Frame {frame_index} could not be read.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        if self.transform:
            frame = self.transform(frame)
        return frame, frame_index

def get_video_frames(args, resize):
    logging.info(f"Preparing to load video frames with resize: {resize}")
    transform = transforms.Compose([
        transforms.Resize((resize, resize), antialias=True),
        transforms.ToTensor(),
    ])
    video_dataset = VideoDataset(video_path=args.video_path, transform=transform)
    logging.info(f"VideoDataset created with {len(video_dataset)} frames.")
    return video_dataset, len(video_dataset)

def quantization(data, filter_number, codec, step=4):
    min_, max_ = codec['min'][filter_number], codec['max'][filter_number]
    data = (data - min_) / (max_ - min_)
    data = data * 255
    data = data.type(dtype=torch.uint8)
    
    quantization_step = step
    data = data / quantization_step
    data = data.type(dtype=torch.uint8)

    return data

def quantization_and_dequantization(data, filter_number, codec, step = 4):
    min_, max_ = codec['min'][filter_number], codec['max'][filter_number]
    
    data = (data - min_) / (max_ - min_)
    data = data * 255
    data = data.type(dtype=torch.uint8)
    
    quantization_step = step
    data = data / quantization_step
    data = data.type(dtype=torch.uint8)
    data = data * quantization_step

    data = data / 255.0
    data = data * (max_ - min_) + min_
    return data

def quantization_and_huffman(data, filter_number, codec, step=4):
    data = data.reshape(-1)
    quantized_data = quantization(data, filter_number, codec, step).cpu().numpy()
    codec = codec['codec'][filter_number]
    encoded = codec.encode(quantized_data)
    return len(encoded) / 1024

def dehuffman_and_dequantization(encoded_data, filter_number, codec, shape, step=4):
    codec_huffman = codec['codec'][filter_number]
    decoded_data = codec_huffman.decode(encoded_data)
    decoded_data = np.array(decoded_data).reshape(shape)
    decoded_data = torch.from_numpy(decoded_data).to('cuda')
    
    min_, max_ = codec['min'][filter_number], codec['max'][filter_number]
    data = decoded_data * step
    data = data.type(dtype=torch.float32)
    data = data / 255.0
    data = data * (max_ - min_) + min_
    return data

def psnr_batch(img1, img2):
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    psnr_values = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_values

def ms_ssim_to_db(ms_ssim):
    return -10 * np.log10(1 - ms_ssim)

def ms_ssim_batch(img1, img2, data_range=1.0):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range)
    
    ms_ssim_values = [ms_ssim(img1[i].unsqueeze(0), img2[i].unsqueeze(0)).item() for i in range(img1.size(0))]
    ms_ssim_db_values = [ms_ssim_to_db(value) for value in ms_ssim_values]
    
    return ms_ssim_db_values

def save_image(img, path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, np.transpose(img, (1, 2, 0)))

def save_latent_images(latent_tensor, path):
    """
    Saves the 12 latent images (each 28x28) from a tensor of size (12, 28, 28) in a single plot.
    
    Parameters:
        latent_tensor (torch.Tensor): The input tensor of size (12, 28, 28).
        path (str): The path to save the output plot.
    """
    # Check if the tensor is on the GPU and move it to the CPU if necessary
    if latent_tensor.is_cuda:
        latent_tensor = latent_tensor.cpu()
    
    # Convert the tensor to numpy for plotting
    latent_images = latent_tensor.detach().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()
    
    for i in range(12):
        ax = axes[i]
        ax.imshow(latent_images[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i+1}')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
def eval_model(model, used_filter, test_dataset, batch_size, codec, output_dir, output_video_dir):
    size_list = []
    psnr_list = []
    ms_ssim_list = []
    all_feature_sizes = []

    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1)

    # Commenting out video writer initialization as decoding is not needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out_path = os.path.join(output_video_dir, f'decode_with_{used_filter}_filters.mp4')
    video_writer = None

    # Create a directory for saving encoded2 data
    encoded2_dir = os.path.join(output_dir, "encoded2")
    os.makedirs(encoded2_dir, exist_ok=True)

    for frame_index, data in enumerate(test_loader):
        images, indices = data
        images = images.to('cuda')
        # encoded = encode_TFLite("MCUCoder.tflite", images)
        # Encoding the video frames
        encoded = model.encoder(images)
        
        model.replace_value = 0
        encoded = model.rate_less(encoded)
        encoded2 = encoded.clone()
        for j in range(used_filter):
            # Quantize and dequantize the encoded data
            encoded[0, j] = quantization_and_dequantization(encoded[0, j], j, codec) 
            
            # Huffman encode the quantized data
            quantized_data = quantization(encoded2[0, j], j, codec).cpu().numpy()
            huffman_encoded = codec['codec'][j].encode(quantized_data.flatten())

            # Save the Huffman-encoded data for each filter
            filter_dir = os.path.join(encoded2_dir, f"filter_{j}")
            os.makedirs(filter_dir, exist_ok=True)
            encoded2_path = os.path.join(filter_dir, f"frame_{frame_index:04d}.bin")
            with open(encoded2_path, "wb") as f:
                f.write(huffman_encoded)

        # Commenting out decoding and video writing
        outputs = model.decoder(encoded)
        os.makedirs(output_dir, exist_ok=True)
        if video_writer is None:
            h, w = outputs[0].shape[1:3]
            video_writer = cv2.VideoWriter(video_out_path, fourcc, 20, (w, h))
        
        output_frame = (outputs[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(output_frame)

        feature_map_size_list = []
        for feature_map in range(used_filter):
            data_size = quantization_and_huffman(encoded[0][feature_map], feature_map, codec)
            feature_map_size_list.append(data_size)

        size_list.append(np.sum(feature_map_size_list))

        # Commenting out saving decoded frames
        if frame_index in [0, 50, 100]:
            save_image(outputs[0].to('cpu').detach().numpy(), os.path.join(output_dir, f"decoded_frames_samples/frame_index_{frame_index}_decoded_with_{int(model.p*12)}_filters.png"))

        # Commenting out PSNR and MS-SSIM calculations as decoding is skipped
        psnr_values = psnr_batch(outputs.detach().cpu(), images.detach().cpu())
        psnr_list.extend(psnr_values)
        
        ms_ssim_values = ms_ssim_batch(outputs.detach().cpu(), images.detach().cpu())
        ms_ssim_list.extend(ms_ssim_values)

    # Commenting out video writer release
    video_writer.release()

    return size_list, psnr_list, ms_ssim_list

def create_codec(test_dataset, model):
    logging.info("Accessing codec creation...")

    codec_setting = {
        'min': {},
        'max': {},
        'codec': {}
    }
    
    temp_loader = DataLoader(test_dataset, batch_size=5000, num_workers=1)
    logging.info("Before 1 codec creation...")
    images, labels = next(iter(temp_loader))
    logging.info("Before 2 codec creation...")
    images = images.to('cuda')

    logging.info("Starting codec creation...")
    for i in tqdm(range(12)):
        # encoded = encode_TFLite("MCUCoder.tflite", images)
        logging.info(f"Creating codec for filter {i + 1}/12...")
        encoded = model.encoder(images)
        data = encoded[:, i, :, :].reshape(-1).detach().clone()
        min_, max_ = torch.min(data), torch.max(data)
        data = ((data - min_) / (max_ - min_) * 255).type(torch.uint8)
        data = (data / 4).type(torch.uint8).cpu().numpy()

        for j in range(0, 63):
            if j not in data:
                data = np.append(data, j)

        codec = HuffmanCodec.from_data(data)
        codec_setting['min'][i], codec_setting['max'][i], codec_setting['codec'][i] = min_, max_, codec
        logging.info(f"Codec for filter {i + 1} created successfully.")

    del temp_loader
    logging.info("Codec creation completed.")
    return codec_setting

def decode_video(encoded2_dir, codec, model, output_video_path, used_filter):

    logging.info(f"Decoding video from {encoded2_dir} using {used_filter} filters...")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    # Get the list of encoded frames
    frame_files = sorted(os.listdir(os.path.join(encoded2_dir, f"filter_0")))
    num_frames = len(frame_files)

    for frame_index in range(num_frames):
        decoded_filters = []
        for j in range(used_filter):
            # Read the Huffman-encoded data
            encoded_file_path = os.path.join(encoded2_dir, f"filter_{j}", f"frame_{frame_index:04d}.bin")
            with open(encoded_file_path, "rb") as f:
                encoded_data = f.read()

            # Decode and dequantize the data
            shape = (28, 28)  # Assuming the latent tensor shape is (28, 28)
            decoded_filter = dehuffman_and_dequantization(encoded_data, j, codec, shape)
            decoded_filters.append(decoded_filter)

        # Pad the latent tensor to have 12 channels
        while len(decoded_filters) < 12:
            decoded_filters.append(torch.zeros_like(decoded_filters[0]))

        # Stack the decoded filters to form the latent tensor
        latent_tensor = torch.stack(decoded_filters, dim=0).unsqueeze(0)

        # Decode the latent tensor to reconstruct the frame
        reconstructed_frame = model.decoder(latent_tensor)

        # Convert the frame to a format suitable for saving
        output_frame = (reconstructed_frame[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        # Initialize the video writer if not already done
        if video_writer is None:
            h, w = output_frame.shape[:2]
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 20, (w, h))

        # Write the frame to the video
        video_writer.write(output_frame)

    # Release the video writer
    video_writer.release()
    logging.info(f"Decoded video saved to {output_video_path}.")

def main():
    args = parse_args()
    torch.manual_seed(0)
    random.seed(10)
    np.random.seed(0)

    logging.info("Initializing the model...")
    model = MCUCoder()
    state_dict = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(state_dict)
    model = model.to('cuda')
    model.eval()
    logging.info("Model loaded and set to evaluation mode.")

    logging.info("Loading the dataset...")
    test_dataset, number_of_frames = get_video_frames(args, 224)
    logging.info(f"Loaded dataset with {number_of_frames} frames.")

    logging.info("Creating codec...")
    codec = create_codec(test_dataset, model)
    logging.info("Codec created successfully.")

    colors = plt.cm.viridis(np.linspace(0, 1, 12))[::-1]  # Use a colormap to get different intensities of the same color

    plt.figure(figsize=(7.5, 7.5 / 1.5))

    for used_filter in range(1, 13):
        logging.info(f"Evaluating model with {used_filter} filters...")
        model.p = used_filter / 12
        frames_sizes, frames_psnrs, frames_ms_ssims = eval_model(
            model,
            used_filter=used_filter,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            codec=codec,
            output_dir=args.output_dir,
            output_video_dir=args.output_dir,
        )

        # Convert image size to bits
        frames_sizes = [size * 8 * 1024 for size in frames_sizes]
        frames_bpps = [size / (224 * 224) for size in frames_sizes]

        avg_frames_bpps = np.mean(frames_bpps)
        avg_frames_psnrs = np.mean(frames_psnrs)
        avg_frames_ms_ssims = np.mean(frames_ms_ssims)

        logging.info(
            f"Filter {used_filter}: bpp={avg_frames_bpps:.4f}, PSNR={avg_frames_psnrs:.2f}dB, MS-SSIM={avg_frames_ms_ssims:.2f}dB"
        )

        # Plot frame index vs bpp
        plt.subplot(3, 1, 1)
        plt.plot(range(len(frames_bpps)), frames_bpps, label=f'[0:{used_filter}]', color=colors[used_filter - 1])
        plt.ylabel('bpp')
        plt.legend(ncol=4, fontsize="6", loc="lower right")

        # Plot frame index vs PSNR
        plt.subplot(3, 1, 2)
        plt.plot(range(len(frames_psnrs)), frames_psnrs, label=f'{used_filter}Filter', color=colors[used_filter - 1])
        plt.ylabel('PSNR')

        # Plot frame index vs MS-SSIM
        plt.subplot(3, 1, 3)
        plt.plot(range(len(frames_ms_ssims)), frames_ms_ssims, label=f'{used_filter}', color=colors[used_filter - 1])
        plt.xlabel('Frame Index')
        plt.ylabel('MS-SSIM')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.savefig('psnr_msssim_per_frame.pdf', bbox_inches='tight')
    plt.show()

    logging.info("Evaluation completed and results saved.")

    # args = parse_args()
    # torch.manual_seed(0)
    # random.seed(10)
    # np.random.seed(0)

    # logging.info("Initializing the model...")
    # model = MCUCoder()
    # state_dict = torch.load(args.model_path, map_location='cuda')
    # model.load_state_dict(state_dict)
    # model = model.to('cuda')
    # model.eval()
    # logging.info("Model loaded and set to evaluation mode.")

    # logging.info("Loading the dataset...")
    # test_dataset, number_of_frames = get_video_frames(args, 224)
    # logging.info(f"Loaded dataset with {number_of_frames} frames.")

    # logging.info("Creating codec...")
    # codec = create_codec(test_dataset, model)
    # logging.info("Codec created successfully.")
    # used_filter = 1
    # # Decode the video
    # encoded2_dir = os.path.join(args.output_dir, "encoded2")
    # output_video_path = os.path.join(args.output_dir, f"decoded2/decoded_use_{used_filter}_filter_video.mp4")
    # decode_video(encoded2_dir, codec, model, output_video_path, used_filter)

    # logging.info("Decoding completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)