# streamlit_wss_app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import map_coordinates
from PIL import Image
import io
import tempfile
import time
from scipy.signal import find_peaks
from itertools import groupby
from operator import itemgetter

# --- Parameters ---
mu = 0.0035
pixel_size_m = 1e-4
resize_scale = 0.5
n_angles = 360
r_max = 30
n_segments = 18
frame_rate = 30.0  # frames per second

# --- Red Mask Function ---
def extract_red_mask(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2) > 0

# --- Frame Extraction ---
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
        frames.append(frame)
    cap.release()
    return frames

# --- WSS Calculation ---
def calculate_wss(frames, progress_callback=None):
    gray_frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (0, 0), fx=resize_scale, fy=resize_scale)
                   for f in frames]
    wss_maps = []
    centers = []
    total = len(gray_frames) - 1
    for i in range(total):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        mask = extract_red_mask(rgb)
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
        if progress_callback:
            progress_callback((i + 1) / total)
    return wss_maps, centers

# --- Wall WSS Time Series ---
def plot_wall_wss_time_series(wss_maps, frames):
    mean_wss_wall = []
    for i, wss in enumerate(wss_maps):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        mask = extract_red_mask(rgb)
        mask_small = cv2.resize(mask.astype(np.uint8), (wss.shape[1], wss.shape[0]))
        contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vals = []
        for c in contours:
            for pt in c:
                x, y = pt[0]
                if 0 <= x < wss.shape[1] and 0 <= y < wss.shape[0]:
                    vals.append(wss[y, x])
        mean_wss_wall.append(np.mean(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mean_wss_wall, marker='o', color='orange')
    ax.set_xlabel("Time [frames]")
    ax.set_ylabel("Mean WSS on Wall [Pa]")
    ax.set_title("Wall Shear Stress Along Vessel Wall Over Time")
    ax.grid(True)
    return fig, mean_wss_wall

# --- Bull's Eye Plot ---
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

# --- Peak grouping to time ranges ---
def format_peak_ranges(peaks):
    if len(peaks) == 0:
        return ""
    start = peaks[0] / frame_rate
    end = peaks[-1] / frame_rate
    return f"{start:.2f}s～{end:.2f}s"

# --- Streamlit UI ---
st.set_page_config(page_title="WSS Short-Axis Analysis", layout="wide")
st.title("🧠 Wall Shear Stress Analysis from Short-Axis Video")

video_file = st.file_uploader("Upload Short-Axis Video (e.g. MP4)", type=["mp4"])

if video_file:
    progress_bar = st.progress(0)
    with st.spinner("Processing video and computing WSS..."):
        frames = extract_frames(video_file)
        wss_maps, centers = calculate_wss(frames, progress_callback=progress_bar.progress)

        st.subheader("1️⃣ Wall WSS Map GIF (flow_shear_masked_shortaxis.gif)")
        st.caption("各フレームで血管壁におけるせん断応力の分布をカラーマップで表示しています。")
        gif_frames = []
        for wss in wss_maps:
            norm = cv2.normalize(wss, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            rgb_img = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
            gif_frames.append(Image.fromarray(rgb_img))

        gif_path = "flow_shear_masked_shortaxis.gif"
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=120, loop=0)
        st.image(gif_path)

        st.subheader("2️⃣ Wall WSS Time Series Plot")
        st.caption("全体の時間軸で、平均WSSの変化を可視化します。血流動態の変動を見るのに有効です。")
        fig1, mean_wss_wall = plot_wall_wss_time_series(wss_maps, frames)
        st.pyplot(fig1)
        max_val = np.max(mean_wss_wall)
        min_val = np.min(mean_wss_wall)
        max_idx = np.argmax(mean_wss_wall)
        peaks, _ = find_peaks(mean_wss_wall, height=np.mean(mean_wss_wall) + np.std(mean_wss_wall))
        peak_range = format_peak_ranges(peaks)
        st.markdown(f"**Highest WSS:** {max_val:.2f} Pa at frame {max_idx} / **Lowest WSS:** {min_val:.2f} Pa")
        if peak_range:
            st.info(f"🟠 WSSが最も高いのは frame {max_idx}（{max_val:.1f} Pa）です。高値は次の時間帯でも見られます：{peak_range}。")

        st.subheader("3️⃣ Bull's Eye Polar Map (18 Segments)")
        st.caption("方向ごとの平均WSSを極座標で可視化。血流の偏りや集中領域を見つけるための指標です。")
        fig2, sector_means, angle_labels = bullseye_map(wss_maps, centers)
        st.pyplot(fig2)
        highest_idx = int(np.argmax(sector_means))
        highest_val = np.max(sector_means)
        st.markdown(f"**Highest segment:** {angle_labels[highest_idx]} with average WSS = **{highest_val:.2f} Pa**")
        st.info(f"🔴 WSSが最も高かったのは {angle_labels[highest_idx]} 方向です。血流が集中している可能性があります。")

    st.success("Analysis complete!")
