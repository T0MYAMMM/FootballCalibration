import os 
import json
import random
import numpy as np
import pandas as pd
import cv2 as cv

from tqdm import tqdm
from PIL import Image

from src.soccerpitch import SoccerPitch
from src.detect_extremities import SegmentationNetwork

lines_palette = [0, 0, 0]
for line_class in SoccerPitch.lines_classes:
    lines_palette.extend(SoccerPitch.palette[line_class])

calibration_folder = "SoccerNet/calibration/test"
clips_folder = "../clips/08fd33_4.mp4"
segmentation_folder = "SoccerNet/calibration/segmentation"


# Ganti dengan path ke file video yang ingin Anda proses
video_path = "SoccerNet/calibration/video/0a2d9b_1.mp4"
cap = cv.VideoCapture(video_path)

lines_palette = [0, 0, 0]
for line_class in SoccerPitch.lines_classes:
    lines_palette.extend(SoccerPitch.palette[line_class])

# Baseline 1: extracting extremities of soccer pitch elements
seg_net = SegmentationNetwork(
    "resources/soccer_pitch_segmentation.pth",
    "resources/mean.npy",
    "resources/std.npy"
)

# Membaca informasi tentang video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
frame_rate = int(cap.get(cv.CAP_PROP_FPS))

# Menyimpan hasil segmentasi dalam sebuah video
output_path = "hasil_segmentasi_video.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height), isColor=False)

# Menggunakan tqdm untuk menampilkan progress bar
with tqdm(total=frame_count) as pbar:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        seg_mask = seg_net.analyse_image(frame)
        mask = Image.fromarray(seg_mask.astype(np.uint8))
        mask = mask.convert('P')
        mask.putpalette(lines_palette)

        # Konversi mask ke dalam format yang dapat ditulis oleh OpenCV
        mask_cv = np.array(mask)

        # Tulis frame segmentasi ke dalam video hasil
        out.write(mask_cv)

        pbar.update(1)  # Update progress bar

# Tutup video hasil
out.release()
cap.release()

print("Video hasil segmentasi telah disimpan di:", output_path)