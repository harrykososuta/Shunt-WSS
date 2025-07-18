# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import map_coordinates
from PIL import Image
import tempfile
from scipy.signal import find_peaks, stft
import pandas as pd

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

def summarize_case(wss, pressure):
    high_wss_threshold = np.mean(wss) + np.std(wss)
    high_pressure_threshold = np.mean(pressure) + np.std(pressure)
    high_wss_ratio = np.sum(wss > high_wss_threshold) / len(wss)
    high_pressure_ratio = np.sum(pressure > high_pressure_threshold) / len(pressure)

    if high_wss_ratio > 0.15 and high_pressure_ratio > 0.15:
        comment = "狭窄の疑いが強い"
    elif high_wss_ratio > 0.15:
        comment = "WSSの負荷が局所的に集中"
    elif high_pressure_ratio > 0.15:
        comment = "血管抵抗の上昇の可能性"
    else:
        comment = "大きな異常は見られない"

    return round(np.max(wss), 1), round(np.max(pressure), 1), round(high_wss_ratio * 100, 1), round(high_pressure_ratio * 100, 1), comment

# --- Streamlit UI ---
st.set_page_config(page_title="Vessel Wall Dynamics Analyzer", layout="wide")
st.title("🤭 Vessel Wall Pressure & Shear Stress Evaluation")

video_file = st.file_uploader("Upload Short-Axis Echo Video (MP4)", type=["mp4"])

def bullseye_map(data_maps, centers, label="WSS"):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    num_rings = 3
    num_sectors = 6
    sector_means = np.random.rand(num_rings * num_sectors)  # 仮データ（将来セグメント平均に置換）
    angle_labels = [f"{i*60}°" for i in range(num_rings * num_sectors)]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))  # サイズ縮小
    width = 2 * np.pi / num_sectors

    for r in range(num_rings):
        inner_radius = r / num_rings
        outer_radius = (r + 1) / num_rings
        for t in range(num_sectors):
            idx = r * num_sectors + t
            theta = t * width
            value = sector_means[idx]
            color = plt.cm.jet(value)
            ax.bar(
                x=theta,
                height=outer_radius - inner_radius,
                width=width,
                bottom=inner_radius,
                color=color,
                edgecolor='white',
                linewidth=1,
                align='edge'
            )

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(f"Bull's Eye ({label})", fontsize=12)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    return fig, sector_means, angle_labels

if video_file:
    st.video(video_file)
    velocity_range = st.slider("速度レンジ（最大血流速度, cm/s）を設定:", min_value=10.0, max_value=120.0, value=50.0, step=1.0)

    if st.button("解析を実行"):
        with st.spinner("Processing video and computing WSS & Pressure..."):
            frames = extract_frames(video_file)
            wss_maps, centers = calculate_wss(frames)
            velocities, pressures = calculate_pressure(frames, velocity_range)
            mean_wss_wall = [np.mean(wss[wss > 0]) for wss in wss_maps]
            time = np.arange(len(pressures)) / frame_rate

            fig1, ax1 = plt.subplots()
            ax1.plot(time, pressures[:len(time)], label="Pressure", color='blue')
            ax1.set_title("Pressure vs Time")
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Pressure")
            ax1.grid(True)

            fig2, ax2 = plt.subplots()
            ax2.plot(time[:len(mean_wss_wall)], mean_wss_wall, color='orange')
            ax2.set_title("WSS vs Time")
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("WSS [Pa]")
            ax2.grid(True)

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            color1 = 'tab:blue'
            color2 = 'tab:orange'
            ax3.set_title("Pressure vs WSS")
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel("Pressure", color=color1)
            ax3.plot(time[:len(mean_wss_wall)], pressures[:len(mean_wss_wall)], color=color1)
            ax3.tick_params(axis='y', labelcolor=color1)

            ax4 = ax3.twinx()
            ax4.set_ylabel("WSS [Pa]", color=color2)
            ax4.plot(time[:len(mean_wss_wall)], mean_wss_wall, color=color2)
            ax4.tick_params(axis='y', labelcolor=color2)
            fig3.tight_layout()

            fig4, sector_means_wss, angle_labels_wss = bullseye_map(wss_maps, centers, label="WSS")
            fig5, sector_means_pressure, angle_labels_pressure = bullseye_map(wss_maps, centers, label="Pressure")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.pyplot(fig2)
            with col2:
                st.pyplot(fig1)
            with col3:
                st.pyplot(fig3)

            col4, col5 = st.columns(2)
            with col4:
                st.pyplot(fig4)
            with col5:
                st.pyplot(fig5)

            st.markdown(f"<div style='text-align:center; font-size:90%; color:gray;'>🔴 WSSが最も高かったのは {angle_labels_wss[np.argmax(sector_means_wss)]} 方向です。</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; font-size:90%; color:gray;'>🔵 Pressureが最も高かったのは {angle_labels_pressure[np.argmax(sector_means_pressure)]} 方向です。</div>", unsafe_allow_html=True)

            st.markdown("---")
             st.subheader("🧐 Summary")
            st.markdown("<div style='background-color: white; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)

            wss_max, p_max, wss_ratio, p_ratio, comment = summarize_case(mean_wss_wall, pressures)
            summary_comment = generate_summary(pressures, mean_wss_wall)
            st.info(summary_comment)

            st.markdown("### 🔶 判読: WSS")
            st.pyplot(fig2)
            st.markdown(f"**Highest WSS:** {np.max(mean_wss_wall):.2f} Pa at frame {np.argmax(mean_wss_wall)}")

            st.markdown("### 🔷 判読: Pressure")
            st.pyplot(fig1)
            st.markdown(f"**Highest Pressure:** {np.max(pressures):.2f} unit at frame {np.argmax(pressures)}")

            st.markdown("### 🔸 判読: WSS と Pressure の関係")
            st.pyplot(fig3)
            st.markdown(f"**WSS and Pressure correlation peak:** frame {np.argmax(np.array(mean_wss_wall) + np.array(pressures))}")

            st.markdown("### 🎯 Bull's Eye Map 判読")
            col4, col5 = st.columns(2)
            with col4:
                st.pyplot(fig4)
                st.markdown(f"**Highest WSS segment:** {angle_labels_wss[np.argmax(sector_means_wss)]} → 平均WSS = {np.max(sector_means_wss):.2f} Pa")
            with col5:
                st.pyplot(fig5)
                st.markdown(f"**Highest Pressure segment:** {angle_labels_pressure[np.argmax(sector_means_pressure)]} → 平均Pressure = {np.max(sector_means_pressure):.2f} unit")

            st.markdown("### 📄 コメント")
            severity_color = "#f28b82" if "狭窄" in comment else ("#fff475" if "可能性" in comment else "#ccff90")
            st.markdown(f"<div style='background-color:{severity_color}; padding:10px; border-radius:8px;'>🗒️ コメント: {comment}</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### ℹ️ パラメータ情報")
            st.markdown(f"- フレーム数: {len(frames)}")
            st.markdown(f"- フレームレート: {frame_rate} fps")
            st.markdown(f"- ピクセルサイズ: {pixel_size_m * 1e4:.2f} μm")
            st.markdown(f"- 血流速度レンジ: {velocity_range:.1f} cm/s")
            st.markdown("</div>", unsafe_allow_html=True)

            summary_df = pd.DataFrame([{
                "WSS最大 [Pa]": wss_max,
                "Pressure最大": p_max,
                "高WSS時間比率 [%]": wss_ratio,
                "高Pressure時間比率 [%]": p_ratio,
                "コメント": comment
            }])

            with st.container():
                st.subheader("📋 結果のCSV出力")
                st.markdown("<div style='background-color: white; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)

                csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button("CSVとして保存", data=csv, file_name="case_summary.csv", mime="text/csv")

                threshold_w = np.mean(mean_wss_wall) + np.std(mean_wss_wall)
                threshold_p = np.mean(pressures) + np.std(pressures)
                peaks_w = sorted(range(len(mean_wss_wall)), key=lambda i: mean_wss_wall[i], reverse=True)[:3]
                peaks_p = sorted(range(len(pressures)), key=lambda i: pressures[i], reverse=True)[:3]

                with st.expander("📸 高WSSが観察されたフレーム"):
                    for idx in peaks_w:
                        st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)

                with st.expander("📸 高Pressureが観察されたフレーム"):
                    for idx in peaks_p:
                        st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)

                suspect_frames = [i for i in range(len(mean_wss_wall))
                                  if pressures[i] > threshold_p and mean_wss_wall[i] > threshold_w]
                if suspect_frames:
                    with st.expander("⚠️ WSSとPressureが同時に高かったフレーム（狭窄の可能性）"):
                        limited_frames = sorted(suspect_frames, key=lambda i: mean_wss_wall[i] + pressures[i], reverse=True)[:5]
                        for idx in limited_frames:
                            st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)
                else:
                    st.info("⚠️ 内圧とWSSが同時に高かったフレームは検出されませんでした。")
                st.markdown("</div>", unsafe_allow_html=True)

            st.success("解析完了！")
