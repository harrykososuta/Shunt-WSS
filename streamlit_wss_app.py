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
    ax.set_title("Bull's Eye WSS Map (18 Segments)", va='bottom')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    plt.tight_layout()
    return fig

# --- Score Summary Helper ---
def generate_score_summary(wss, pressure):
    wss_thresh = np.mean(wss) + np.std(wss)
    pressure_thresh = np.mean(pressure) + np.std(pressure)
    high_wss_ratio = np.sum(wss > wss_thresh) / len(wss)
    high_pressure_ratio = np.sum(pressure > pressure_thresh) / len(pressure)
    if high_wss_ratio > 0.15 and high_pressure_ratio > 0.15:
        score = 5
        comment = "高度狭窄の疑い"
    elif high_wss_ratio > 0.15 or high_pressure_ratio > 0.15:
        score = 3
        comment = "中等度狭窄の可能性"
    else:
        score = 1
        comment = "狭窄の可能性は低い"
    return score, comment

# --- UI Layout ---
st.set_page_config(page_title="Vessel Wall Dynamics Analyzer", layout="wide")
st.title("🧐 Vessel Wall Pressure & Shear Stress Evaluation")

video_file = st.file_uploader("Upload Short-Axis Echo Video (MP4)", type=["mp4"])

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

            # グラフ
            fig1, ax1 = plt.subplots()
            ax1.plot(time, pressures[:len(time)], label="Pressure", color='blue')
            ax1.set_xlabel("Time [s]")
            ax1.set_ylabel("Pressure")
            ax1.set_title("Pressure vs Time")

            fig2, ax2 = plt.subplots()
            ax2.plot(time[:len(mean_wss_wall)], mean_wss_wall, label="WSS", color='orange')
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("WSS [Pa]")
            ax2.set_title("WSS vs Time")

            fig3, ax3 = plt.subplots()
            ax3.plot(time[:len(mean_wss_wall)], pressures[:len(mean_wss_wall)], color='blue')
            ax3.set_ylabel("Pressure", color='blue')
            ax4 = ax3.twinx()
            ax4.plot(time[:len(mean_wss_wall)], mean_wss_wall, color='orange')
            ax4.set_ylabel("WSS [Pa]", color='orange')
            ax3.set_xlabel("Time [s]")
            ax3.set_title("WSS vs Pressure")
            fig3.tight_layout()

            col1, col2, col3 = st.columns(3)
            with col1: st.pyplot(fig2)
            with col2: st.pyplot(fig1)
            with col3: st.pyplot(fig3)

            # Bull's Eye
            fig4 = bullseye_map(wss_maps, centers)
            col4, col5 = st.columns([2, 1])
            with col4:
                st.pyplot(fig4)
            with col5:
                st.markdown("""
                ### WSSとは？
                血管内皮細胞にかかるずり応力。高いと内皮障害やリモデリングに関連。

                ### Pressureとは？
                血管内腔の内圧を表す指標。上昇は抵抗増加や狭窄を示唆。
                """)
                with st.expander("🧠 医工学的な重要ポイントの解説"):
                    st.markdown("""
                    - **内圧とWSSが同時に上昇する時間帯**は、**狭窄や血流の局所集中が疑われる重要ポイント**です。
                    - **WSSのみが上昇している場合**は、血流が局所的に偏っており、血管壁への**摩染的ストレスが増大**していることを示します。
                    - **内圧のみ上昇している場合**は、血管壁の**弾性低下や外的圧迫**の可能性があります。
                    """)

            # Summary Score
            with st.container():
                st.subheader("📊 Summary & Score")
                score, comment = generate_score_summary(mean_wss_wall, pressures)
                st.markdown(f"**Score:** {score} / 5")
                st.info(comment)

            # 結果CSV & 高値フレーム
            threshold_w = np.mean(mean_wss_wall) + np.std(mean_wss_wall)
            threshold_p = np.mean(pressures) + np.std(pressures)

            results_df = pd.DataFrame({
                "Time [s]": time[:len(mean_wss_wall)],
                "Pressure [arb. unit]": pressures[:len(mean_wss_wall)],
                "Mean WSS [Pa]": mean_wss_wall
            })

            with st.container():
                st.subheader("⬇️ 結果のCSV出力")
                st.download_button("📄 CSVをダウンロード", data=results_df.to_csv(index=False), file_name="WSS_vs_Pressure_Output.csv", mime="text/csv")

                with st.expander("📸 高WSSが観察されたフレーム"):
                    peaks, _ = find_peaks(mean_wss_wall, height=threshold_w)
                    top_wss_frames = sorted(peaks, key=lambda i: mean_wss_wall[i], reverse=True)[:3]
                    for idx in top_wss_frames:
                        st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)

                with st.expander("📸 高Pressureが観察されたフレーム"):
                    peaks_p, _ = find_peaks(pressures, height=threshold_p)
                    top_pressure_frames = sorted(peaks_p, key=lambda i: pressures[i], reverse=True)[:3]
                    for idx in top_pressure_frames:
                        st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)

                with st.expander("⚠️ WSSとPressureが同時に高かったフレーム（狭窄の可能性）"):
                    both_high = [i for i in range(len(mean_wss_wall)) if mean_wss_wall[i] > threshold_w and pressures[i] > threshold_p]
                    top_both = sorted(both_high, key=lambda i: mean_wss_wall[i] + pressures[i], reverse=True)[:3]
                    if top_both:
                        for idx in top_both:
                            st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)
                    else:
                        st.info("該当するフレームは検出されませんでした。")
