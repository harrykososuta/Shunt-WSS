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

# --- 関数セット ---
def extract_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower1, upper1 = np.array([0,70,50]), np.array([10,255,255])
    lower2, upper2 = np.array([160,70,50]), np.array([180,255,255])
    return ((cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)) > 0)

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

def calculate_wss(frames):
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (0,0),
                       fx=resize_scale, fy=resize_scale) for f in frames]
    wss_maps, centers = [], []
    for i in range(len(gray)-1):
        mask = extract_red_mask(frames[i])
        small_mask = cv2.resize(mask.astype(np.uint8),
                                (gray[i].shape[1], gray[i].shape[0])) > 0
        coords = np.column_stack(np.where(mask))
        cy, cx = coords.mean(axis=0).astype(int)
        centers.append((cx, cy))
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i+1], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        du = cv2.Sobel(flow[...,0], cv2.CV_32F, 1, 0, 3)
        dv = cv2.Sobel(flow[...,1], cv2.CV_32F, 0, 1, 3)
        mag = np.sqrt(du**2 + dv**2)
        wss_maps.append(np.where(small_mask, mu * mag / pixel_size_m, 0))
    return wss_maps, centers

def calculate_pressure(frames, vmax):
    red_means = [(frame[...,0][extract_red_mask(frame)].mean()
                  if extract_red_mask(frame).any() else 0)
                  for frame in frames]
    M = max(red_means) or 1
    velocities = [(m / M) * vmax for m in red_means]
    pressures = [np.pi * (0.25)**2 * v for v in velocities]
    return velocities, pressures

def bullseye_map_highlight(values, title, cmap='jet', highlight_only=True):
    sectors = 12
    vals = np.array(values)
    if vals.size < sectors:
        vals = np.pad(vals, (0, sectors - vals.size),
                      constant_values=np.nan)
    threshold = np.nanmean(vals) + np.nanstd(vals)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4, 4))
    width = 2 * np.pi / sectors
    inner_r, outer_r = 0.8, 1.0
    for i in range(sectors):
        theta = i * width
        val = vals[i]
        if np.isnan(val):
            color = 'lightgrey'
        elif highlight_only and val < threshold:
            color = 'white'
        else:
            norm = (val - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-6)
            color = plt.get_cmap(cmap)(norm)
        ax.bar(theta, outer_r - inner_r, width=width,
               bottom=inner_r, color=color,
               edgecolor='black', linewidth=0.8)
    ax.set_xticks(np.linspace(0, 2*np.pi, sectors, endpoint=False))
    ax.set_xticklabels([f"{i*30}°" for i in range(sectors)])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return fig, vals

def generate_summary_text(time, pressures, wss):
    t_p = time[np.argmax(pressures)]
    t_w = time[np.argmax(wss)]
    text_w = f"【WSS結果】{t_w:.2f} 秒 に最大（{np.max(wss):.2f} Pa）を示しました。"
    text_p = f"【Pressure結果】{t_p:.2f} 秒 に最大（{np.max(pressures):.2f}）を示しました。"
    return text_w, text_p

def summarize_case(wss, pressure):
    thr_w = np.mean(wss) + np.std(wss)
    thr_p = np.mean(pressure) + np.std(pressure)
    hw = np.sum(np.array(wss) > thr_w) / len(wss)
    hp = np.sum(np.array(pressure) > thr_p) / len(pressure)
    if hw > 0.15 and hp > 0.15:
        comment = "狭窄の疑いが強い"
    elif hw > 0.15:
        comment = "WSSに局所負荷が集中"
    elif hp > 0.15:
        comment = "血管抵抗増加の可能性"
    else:
        comment = "異常なし"
    return round(np.max(wss), 1), round(np.max(pressure), 1), round(hw * 100, 1), round(hp * 100, 1), comment

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

            # グラフ作成
            fig_w, axw = plt.subplots()
            axw.plot(time[:len(mean_wss)], mean_wss, color='orange')
            axw.set_title("WSS vs Time"); axw.set_xlabel("Time [s]")

            fig_p, axp = plt.subplots()
            axp.plot(time, pressures, color='blue')
            axp.set_title("Pressure vs Time"); axp.set_xlabel("Time [s]")

            fig_pw, axpw = plt.subplots()
            axpw.plot(time[:len(mean_wss)], pressures[:len(mean_wss)], 'b-')
            axpw2 = axpw.twinx()
            axpw2.plot(time[:len(mean_wss)], mean_wss, 'orange')
            axpw.set_title("WSS vs Pressure"); axpw.set_xlabel("Time [s]")

            fig_be_w, vals_w = bullseye_map_highlight(mean_wss[:12], "Bull’s Eye (WSS)", cmap='Blues')
            fig_be_p, vals_p = bullseye_map_highlight(np.array(pressures[:12]), "Bull’s Eye (Pressure)", cmap='Reds')

            # レイアウト表示
            st.subheader("📈 計測グラフ")
            c1, c2, c3 = st.columns(3)
            c1.pyplot(fig_w); c2.pyplot(fig_p); c3.pyplot(fig_pw)

            st.subheader("🎯 Bull’s Eye Map")
            b1, b2 = st.columns(2)
            b1.pyplot(fig_be_w); b2.pyplot(fig_be_p)

            st.markdown("### 🧠 Summary")
            st.markdown("<div style='background-color: white; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)

            # 解説ボタン
            with st.expander("WSSとPressureの説明"):
                st.markdown("""
- **WSS（Wall Shear Stress）**：血管壁にかかるずり応力のことで、高いと血管障害の可能性があります。
- **Pressure（内圧）**：模擬的に算出した血管内部の圧力推移であり、高い値は血流抵抗や壁応力を示唆します。
""")

            # 簡易結果文
            tw, tp = generate_summary_text(time, pressures, mean_wss)
            st.markdown(f"- {tw}")
            st.markdown(f"- {tp}")

            # スコア
            wss_max, p_max, wsr, pr, comment = summarize_case(mean_wss, pressures)
            st.markdown("### 📊 スコア結果")
            st.markdown(f"・最大WSS：**{wss_max} Pa**  \n・最大Pressure：**{p_max}**  \n・高WSS時間比率：**{wsr}%**  \n・高Pressure時間比率：**{pr}%**")

            # 総合判定
            severity = ("#d1e7dd" if "異常なし" in comment
                        else "#fff3cd" if "可能性" in comment
                        else "#f8d7da")
            st.markdown(f"<div style='background-color:{severity}; color:#000; padding:15px; border-radius:10px; text-align:center; font-weight:bold;'>総合判定：{comment}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # CSV
            st.subheader("📋 結果CSV")
            df = pd.DataFrame([{"WSS最大 [Pa]": wss_max, "Pressure最大": p_max,
                                "高WSS時間比率 [%]": wsr, "高Pressure時間比率 [%]": pr,
                                "総合コメント": comment}])
            st.download_button("CSVとして保存", data=df.to_csv(index=False).encode(), file_name="summary.csv")

            # 高値フレーム
            st.subheader("📸 高値フレーム表示")
            thr_w = np.nanmean(mean_wss) + np.nanstd(mean_wss)
            thr_p = np.nanmean(pressures) + np.nanstd(pressures)
            peaks_w = np.argsort(mean_wss)[-3:][::-1]
            peaks_p = np.argsort(pressures)[-3:][::-1]

            with st.expander("高WSSフレーム"):
                for i in peaks_w:
                    st.image(frames[i], caption=f"Frame {i} – {i/frame_rate:.2f}s")

            with st.expander("高Pressureフレーム"):
                for i in peaks_p:
                    st.image(frames[i], caption=f"Frame {i} – {i/frame_rate:.2f}s")

            with st.expander("WSSとPressureが同時に高いフレーム"):
                suspects = [i for i in range(len(mean_wss)) if mean_wss[i] > thr_w and pressures[i] > thr_p]
                if suspects:
                    for i in suspects[:3]:
                        st.image(frames[i], caption=f"Frame {i} – {i/frame_rate:.2f}s")
                else:
                    st.info("該当フレームはありません。")

            st.success("✅ 解析完了！")
