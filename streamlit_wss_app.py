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

def calculate_pressure(frames):
    mean_velocities = []
    for frame in frames:
        mask = extract_red_mask(frame)
        red_intensity = frame[..., 0]
        mean_value = np.mean(red_intensity[mask]) if np.any(mask) else 0
        mean_velocities.append(mean_value)

    max_red = max(mean_velocities) if max(mean_velocities) > 0 else 1
    velocities = [(v / max_red) * velocity_range for v in mean_velocities]
    A = np.pi * (0.25)**2  # cm^2
    Z = 1.0
    pressures = [Z * A * v for v in velocities]
    return velocities, pressures

def generate_summary(pressures, mean_wss_wall):
    mean_pressure = np.mean(pressures)
    peak_pressure = np.max(pressures)
    mean_wss = np.mean(mean_wss_wall)
    peak_wss = np.max(mean_wss_wall)
    if peak_pressure > 1.2 * mean_pressure and peak_wss > 1.2 * mean_wss:
        return "この箇所に内圧上昇箇所が見られます。WSSも上昇しているので狭窄が疑われます。"
    elif peak_pressure > 1.2 * mean_pressure:
        return "内圧の上昇が観察されましたが、WSSの変化は限定的です。弾性変化の可能性があります。"
    elif peak_wss > 1.2 * mean_wss:
        return "WSSの上昇が観察されました。流速分布の局所的集中により負荷がかかっている可能性があります。"
    else:
        return "顕著な内圧やWSSの異常は観察されませんでした。"

# --- Streamlit UI ---
st.set_page_config(page_title="Vessel Wall Dynamics Analyzer", layout="wide")
st.title("🧠 Vessel Wall Pressure & Shear Stress Evaluation")

video_file = st.file_uploader("Upload Short-Axis Echo Video (MP4)", type=["mp4"])

if video_file:
    with st.spinner("Processing video and computing WSS & Pressure..."):
        frames = extract_frames(video_file)
        wss_maps, centers = calculate_wss(frames)
        velocities, pressures = calculate_pressure(frames)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        time = np.arange(len(pressures)) / frame_rate
        ax1.plot(time, pressures[:len(time)], label="Pressure", color='blue')
        ax1.set_title("Estimated Central Pressure Over Time")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Pressure [arb. unit]")
        ax1.grid(True)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(mean_wss_wall := [np.mean(wss[wss > 0]) for wss in wss_maps], color='orange', marker='o')
        ax2.set_title("Wall Shear Stress Along Vessel Wall Over Time")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Mean WSS [Pa]")
        ax2.grid(True)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig2)
        with col2:
            st.pyplot(fig1)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(time[:len(mean_wss_wall)], pressures[:len(mean_wss_wall)], label="Pressure", color='blue')
        ax3.plot(time[:len(mean_wss_wall)], mean_wss_wall, label="WSS", color='orange')
        ax3.set_xlabel("Time [s]")
        ax3.set_title("Pressure vs WSS")
        ax3.legend()
        ax3.grid(True)

        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(fig3)

        summary = generate_summary(pressures, mean_wss_wall)
        with col4:
            st.subheader("💡 Summary")
            st.info(summary)

        max_val = np.max(mean_wss_wall)
        min_val = np.min(mean_wss_wall)
        max_idx = np.argmax(mean_wss_wall)
        peaks, _ = find_peaks(mean_wss_wall, height=np.mean(mean_wss_wall) + np.std(mean_wss_wall))
        peak_range = f"{peaks[0]/frame_rate:.2f}s～{peaks[-1]/frame_rate:.2f}s" if len(peaks) > 0 else ""

        st.markdown(f"**Highest WSS:** {max_val:.2f} Pa at frame {max_idx} / **Lowest WSS:** {min_val:.2f} Pa")
        if peak_range:
            st.info(f"🟠 WSSが最も高いのは frame {max_idx}（{max_val:.1f} Pa）です。高値は次の時間帯でも見られます：{peak_range}。")

        st.success("Analysis complete.")
