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

def calculate_pressure(frames, velocity_range):
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
st.title("🧐 Vessel Wall Pressure & Shear Stress Evaluation")

video_file = st.file_uploader("Upload Short-Axis Echo Video (MP4)", type=["mp4"])

if video_file:
    st.video(video_file)
    velocity_range = st.slider(
        "速度レンジ（最大血流速度, cm/s）を設定:",
        min_value=10.0,
        max_value=200.0,
        value=76.0,
        step=1.0,
        help="血流速度の最大値を調整してください。"
    )

    if st.button("解析を実行"):
        with st.spinner("Processing video and computing WSS & Pressure..."):
            frames = extract_frames(video_file)
            wss_maps, centers = calculate_wss(frames)
            velocities, pressures = calculate_pressure(frames, velocity_range)

            mean_wss_wall = [np.mean(wss[wss > 0]) for wss in wss_maps]
            time = np.arange(len(pressures)) / frame_rate

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(time, pressures[:len(time)], label="Pressure", color='blue')
            ax1.set_title("Estimated Central Pressure Over Time")
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Pressure [arb. unit]")
            ax1.grid(True)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(mean_wss_wall, color='orange', marker='o')
            ax2.set_title("Wall Shear Stress Along Vessel Wall Over Time")
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Mean WSS [Pa]")
            ax2.grid(True)

            fig4, sector_means, angle_labels = bullseye_map(wss_maps, centers)

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig2)
            with col2:
                st.pyplot(fig1)

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            color1 = 'tab:blue'
            color2 = 'tab:orange'
            ax3.set_title("Pressure vs WSS")
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("Pressure [arb. unit]", color=color1)
            ax3.plot(time[:len(mean_wss_wall)], pressures[:len(mean_wss_wall)], label="Pressure", color=color1)
            ax3.tick_params(axis='y', labelcolor=color1)

            ax4 = ax3.twinx()
            ax4.set_ylabel("WSS [Pa]", color=color2)
            ax4.plot(time[:len(mean_wss_wall)], mean_wss_wall, label="WSS", color=color2)
            ax4.tick_params(axis='y', labelcolor=color2)

            fig3.tight_layout()

            col3, col4 = st.columns(2)
            with col3:
                st.pyplot(fig3)
            with col4:
                st.pyplot(fig4)

            summary = generate_summary(pressures, mean_wss_wall)
            st.subheader("💡 Summary")
            st.info(summary)

            max_val = np.max(mean_wss_wall)
            min_val = np.min(mean_wss_wall)
            max_idx = np.argmax(mean_wss_wall)
            peaks, _ = find_peaks(mean_wss_wall, height=np.mean(mean_wss_wall) + np.std(mean_wss_wall))
            peak_range = f"{peaks[0]/frame_rate:.2f}s〜{peaks[-1]/frame_rate:.2f}s" if len(peaks) > 0 else ""

            st.markdown(f"**Highest WSS:** {max_val:.2f} Pa at frame {max_idx} / **Lowest WSS:** {min_val:.2f} Pa")
            if peak_range:
                st.info(f"🟠 WSSが最も高いのは frame {max_idx}（{max_val:.1f} Pa）です。高値は次の時間帯でも見られます：{peak_range}。")

            highest_idx = int(np.argmax(sector_means))
            highest_val = np.max(sector_means)
            st.markdown(f"**Highest segment:** {angle_labels[highest_idx]} with average WSS = **{highest_val:.2f} Pa**")
            st.info(f"🔴 WSSが最も高かったのは {angle_labels[highest_idx]} 方向です。血流が集中している可能性があります。")

            st.success("Analysis complete.")

            with st.expander("🧠 医工学的な重要ポイントの解説（クリックで展開）"):
                st.markdown("""
- **内圧とWSSが同時に上昇する時間帯**は、**狭窄や血流の局所集中が疑われる重要ポイント**です。
- **WSSのみが上昇している場合**は、血流が局所的に偏っており、血管壁への**摩染的ストレスが増大**していることを示します。
- **内圧のみ上昇している場合**は、血管壁の**弾性低下や外的圧迫**の可能性があり、流速は比較的安定していると考えられます。
""")

            with st.expander("📸 高WSSが観察されたフレーム"):
                top_peaks = sorted(peaks, key=lambda i: mean_wss_wall[i], reverse=True)[:3]
                for idx in top_peaks:
                    st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)

            threshold_p = np.mean(pressures) + np.std(pressures)
            threshold_w = np.mean(mean_wss_wall) + np.std(mean_wss_wall)
            suspect_frames = [i for i in range(len(mean_wss_wall))
                              if pressures[i] > threshold_p and mean_wss_wall[i] > threshold_w]

            if suspect_frames:
                with st.expander("⚠️ WSSとPressureが同時に高かったフレーム（狭窄の可能性）"):
                    limited_frames = sorted(suspect_frames, key=lambda i: mean_wss_wall[i] + pressures[i], reverse=True)[:5]
                    for idx in limited_frames:
                        st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)
            else:
                st.info("⚠️ 内圧とWSSが同時に高かったフレームは検出されませんでした。")
