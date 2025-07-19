# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd

# --- パラメータ ---
mu = 0.0035
pixel_size_m = 1e-4
resize_scale = 0.5
frame_rate = 30.0

# --- Red Mask 抽出 ---
def extract_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower1, upper1 = np.array([0,70,50]), np.array([10,255,255])
    lower2, upper2 = np.array([160,70,50]), np.array([180,255,255])
    return ((cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)) > 0)

# --- フレーム抽出 ---
def extract_frames(video_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_file.read())
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# --- WSS 計算 ---
def calculate_wss(frames):
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (0,0), fx=resize_scale, fy=resize_scale) for f in frames]
    wss_maps, centers = [], []
    for i in range(len(gray)-1):
        mask = extract_red_mask(frames[i])
        small_mask = cv2.resize(mask.astype(np.uint8), (gray[i].shape[1], gray[i].shape[0]))>0
        coords = np.column_stack(np.where(mask))
        cy, cx = coords.mean(axis=0).astype(int)
        centers.append((cx, cy))
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i+1], None, 0.5,3,15,3,5,1.2,0)
        du = cv2.Sobel(flow[...,0],cv2.CV_32F,1,0,3)
        dv = cv2.Sobel(flow[...,1],cv2.CV_32F,0,1,3)
        mag = np.sqrt(du**2 + dv**2)
        wss_maps.append(np.where(small_mask, mu*mag/pixel_size_m, 0))
    return wss_maps, centers

# --- Pressure 計算 ---
def calculate_pressure(frames, vmax):
    red_means = [(frame[...,0][extract_red_mask(frame)].mean() if extract_red_mask(frame).any() else 0) for frame in frames]
    M = max(red_means) or 1
    return [(m/M)*vmax for m in red_means], [np.pi*(0.25)**2*(m/M)*vmax for m in red_means]

# --- Bull’s Eye 表示（外周のみ） ---
def bullseye_map_wall_only(values, title):
    sectors = 12
    vals = np.array(values)
    if vals.size < sectors:
        vals = np.pad(vals, (0, sectors - vals.size), constant_values=np.nan)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4,4))
    width = 2 * np.pi / sectors
    inner_r, outer_r = 0.8, 1.0
    for i in range(sectors):
        theta = i * width
        color_val = np.nan_to_num(vals[i], nan=0.0)
        ax.bar(theta, outer_r - inner_r, width=width,
               bottom=inner_r,
               color=plt.cm.jet(color_val),
               edgecolor='white', linewidth=1)
    ax.set_xticks(np.linspace(0, 2*np.pi, sectors, endpoint=False))
    ax.set_xticklabels([f"{i*30}°" for i in range(sectors)])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return fig, vals

# --- Summary文章 ---
def generate_summary_text(time, pressures, wss):
    t_p = time[np.argmax(pressures)]
    t_w = time[np.argmax(wss)]
    text_p = f"🩸 Pressure は **{t_p:.2f} 秒** に最大（{np.max(pressures):.2f}）を示しました。"
    text_w = f"🌀 WSS は **{t_w:.2f} 秒** に最大（{np.max(wss):.2f} Pa）を示しました。"
    return text_p, text_w

# --- スコアリング関数 ---
def summarize_case(wss, pressure):
    thr_w = np.mean(wss) + np.std(wss)
    thr_p = np.mean(pressure) + np.std(pressure)
    hw = np.sum(np.array(wss) > thr_w) / len(wss)
    hp = np.sum(np.array(pressure) > thr_p) / len(pressure)
    if hw>0.15 and hp>0.15:
        comment="狭窄の疑いが強い"
    elif hw>0.15:
        comment="WSSに局所負荷が集中"
    elif hp>0.15:
        comment="血管抵抗増加の可能性"
    else:
        comment="異常なし"
    return round(np.max(wss),1), round(np.max(pressure),1), round(hw*100,1), round(hp*100,1), comment

# --- Streamlit UI ---
st.set_page_config(page_title="Vessel WSS & Pressure Analyzer", layout="wide")
st.title("🧐 Vessel Wall Pressure & Shear Stress Evaluation")

video_file = st.file_uploader("📤 動画アップロード（MP4）", type="mp4")
if video_file:
    st.video(video_file)
    vmax = st.slider("速度レンジ (最大血流速度 cm/s)", 10.0, 120.0, 50.0)
    if st.button("解析を実行"):
        with st.spinner("🧮 計算中..."):
            frames = extract_frames(video_file)
            wss_maps, centers = calculate_wss(frames)
            velocities, pressures = calculate_pressure(frames, vmax)
            mean_wss = np.array([np.nanmean(w) for w in wss_maps])
            time = np.arange(len(pressures)) / frame_rate

            fig_w, axw = plt.subplots(); axw.plot(time[:len(mean_wss)], mean_wss, color='orange'); axw.set_title("WSS vs Time"); axw.set_xlabel("Time [s]")
            fig_p, axp = plt.subplots(); axp.plot(time, pressures, color='blue'); axp.set_title("Pressure vs Time"); axp.set_xlabel("Time [s]")
            fig_pw, axpw = plt.subplots(); axpw.plot(time[:len(mean_wss)], pressures[:len(mean_wss)], 'b-'); axpw2 = axpw.twinx(); axpw2.plot(time[:len(mean_wss)], mean_wss, 'orange'); axpw.set_title("WSS vs Pressure"); axpw.set_xlabel("Time [s]")

            fig_be_w, vals_w = bullseye_map_wall_only(mean_wss[:12], "Bull’s Eye (WSS)")
            fig_be_p, vals_p = bullseye_map_wall_only(np.array(pressures[:12]), "Bull’s Eye (Pressure)")

            st.subheader("📈 計測グラフ")
            cols = st.columns(3)
            cols[0].pyplot(fig_w)
            cols[1].pyplot(fig_p)
            cols[2].pyplot(fig_pw)

            st.subheader("🎯 Bull’s Eye Map")
            becols = st.columns(2)
            becols[0].pyplot(fig_be_w)
            becols[1].pyplot(fig_be_p)

            st.markdown("---")
            st.subheader("🧠 Summary")
            st.markdown("<div style='background:white;padding:10px;border-radius:8px;'>", unsafe_allow_html=True)

            # Text summary
            txt_p, txt_w = generate_summary_text(time, np.array(pressures), mean_wss)
            st.write(txt_w); st.write(txt_p)

            wss_max, p_max, wsr, pr, comment = summarize_case(mean_wss, pressures)

            st.markdown("### 📊 スコア結果")
            st.write(f"- 最大WSS: **{wss_max} Pa**")
            st.write(f"- 最大Pressure: **{p_max}**")
            st.write(f"- 高WSS時間比率: **{wsr}%**")
            st.write(f"- 高Pressure時間比率: **{pr}%**")
            st.markdown(f"**🩺 総合判定: `{comment}`**")

            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📋 結果CSV")
            df = pd.DataFrame([{
                "WSS最大 [Pa]": wss_max,
                "Pressure最大": p_max,
                "高WSS時間比率 [%]": wsr,
                "高Pressure時間比率 [%]": pr,
                "コメント": comment
            }])
            st.download_button("CSVとして保存", data=df.to_csv(index=False).encode(), file_name="summary.csv")

            # 画像出力
            st.subheader("📸 高値フレーム表示")
            thr_w = np.mean(mean_wss) + np.std(mean_wss)
            thr_p = np.mean(pressures) + np.std(pressures)
            peaks_w = np.argsort(mean_wss)[-3:][::-1]
            peaks_p = np.argsort(pressures)[-3:][::-1]

            with st.expander("🔶 高WSSが観察されたフレーム"):
                for i in peaks_w:
                    st.image(frames[i], caption=f"Frame {i} - {i/frame_rate:.2f}s")

            with st.expander("🔷 高Pressureが観察されたフレーム"):
                for i in peaks_p:
                    st.image(frames[i], caption=f"Frame {i} - {i/frame_rate:.2f}s")

            with st.expander("⚠️ WSSとPressureが同時に高かったフレーム（狭窄疑い）"):
                suspects = [i for i in range(len(mean_wss)) if mean_wss[i]>thr_w and pressures[i]>thr_p]
                if suspects:
                    for i in suspects[:3]:
                        st.image(frames[i], caption=f"Frame {i} - {i/frame_rate:.2f}s")
                else:
                    st.info("同時に高値を示すフレームは検出されませんでした。")

            st.success("解析完了！")
