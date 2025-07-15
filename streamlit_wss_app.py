# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import map_coordinates
from PIL import Image
import tempfile
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# --- Parameters ---
mu = 0.0035
pixel_size_m = 1e-4
resize_scale = 0.5
n_angles = 360
r_max = 30
n_segments = 18
frame_rate = 30.0
velocity_range = 76.0  # cm/s

# --- Utility Functions ---
def extract_red_mask(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return (mask1 | mask2) > 0

def extract_frames(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(video_bytes.read())
        temp_path = tmpfile.name

    cap = cv2.VideoCapture(temp_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def calculate_wss(frames):
    gray_frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (0, 0), fx=resize_scale, fy=resize_scale)
                   for f in frames]
    wss_maps = []
    centers = []
    for i in range(len(gray_frames) - 1):
        mask = extract_red_mask(frames[i])
        mask_small = cv2.resize(mask.astype(np.uint8), (gray_frames[i].shape[1], gray_frames[i].shape[0])) > 0
        coords = np.column_stack(np.where(mask))
        cy, cx = np.mean(coords, axis=0).astype(int)
        cy = int(cy * resize_scale)
        cx = int(cx * resize_scale)
        centers.append((cx, cy))
        flow = cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i + 1], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        du_dx = cv2.Sobel(flow[..., 0], cv2.CV_32F, 1, 0, ksize=3)
        dv_dy = cv2.Sobel(flow[..., 1], cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(du_dx ** 2 + dv_dy ** 2)
        wss_map = mu * grad_mag / pixel_size_m
        wss_masked = np.where(mask_small, wss_map, 0)
        wss_maps.append(wss_masked)
    return wss_maps, centers

def bullseye_map(wss_maps, centers):
    wss_polar = np.zeros((len(wss_maps), n_angles))
    for t, wss in enumerate(wss_maps):
        cx, cy = centers[t]
        for j, theta in enumerate(np.linspace(0, 2 * np.pi, n_angles, endpoint=False)):
            r_vals = np.linspace(5, r_max, num=20)
            x_coords = cx + r_vals * np.cos(theta)
            y_coords = cy + r_vals * np.sin(theta)
            coords = np.vstack([y_coords, x_coords])
            values = map_coordinates(wss, coords, order=1, mode='constant', cval=0.0)
            wss_polar[t, j] = np.nanmean(values)

    sector_angles = np.linspace(0, 360, n_segments + 1)
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    sector_means = []
    angle_labels = []
    for i in range(n_segments):
        indices = np.arange(int(sector_angles[i]), int(sector_angles[i + 1])) % 360
        sector_means.append(np.nanmean(wss_polar[:, indices]))
        angle_labels.append(f"{int(sector_angles[i])}°–{int(sector_angles[i+1])}°")

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    bars = ax.bar(theta, sector_means, width=2 * np.pi / n_segments, bottom=0,
                  color=plt.cm.jet((np.array(sector_means) - np.min(sector_means)) /
                                   (np.max(sector_means) - np.min(sector_means))))
    for i, (angle, val) in enumerate(zip(theta, sector_means)):
        ax.text(angle, val + 0.005, f"{angle_labels[i]}\n{val:.1f}", ha='center', va='bottom', fontsize=7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_title("Bull's Eye WSS Map (18 Segments)", va='bottom')
    plt.tight_layout()
    return fig, sector_means, angle_labels

# --- Streamlit UI ---
st.set_page_config(page_title="Vessel Wall Dynamics Analyzer", layout="wide")
st.title("🧠 Vessel Wall Pressure & Shear Stress Evaluation")

st.markdown("""
このアプリケーションでは、短軸心エコー動画から血管壁に作用する壁せん断応力（WSS）と中心内圧（推定値）を解析し、
その時系列的変化・空間的分布を視覚的に評価することができます。
""")
