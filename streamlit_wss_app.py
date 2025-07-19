# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd

# --- Parameters ---
mu = 0.0035
pixel_size_m = 1e-4
resize_scale = 0.5
frame_rate = 30.0

# --- Utility Functions ---
def extract_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower1, upper1 = np.array([0,70,50]), np.array([10,255,255])
    lower2, upper2 = np.array([160,70,50]), np.array([180,255,255])
    return ((cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)) > 0)

def extract_frames(video_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_file.read()); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def calculate_wss(frames):
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY),
                       (0,0), fx=resize_scale, fy=resize_scale) for f in frames]
    wss_maps = []
    for i in range(len(gray)-1):
        mask = extract_red_mask(frames[i])
        small = cv2.resize(mask.astype(np.uint8),
                           (gray[i].shape[1], gray[i].shape[0])) > 0
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i+1], None,
                                            0.5,3,15,3,5,1.2,0)
        du = cv2.Sobel(flow[...,0], cv2.CV_32F,1,0,3)
        dv = cv2.Sobel(flow[...,1], cv2.CV_32F,0,1,3)
        wss_maps.append(np.where(small,
                                 mu * np.sqrt(du**2 + dv**2) / pixel_size_m, 0))
    return wss_maps

def calculate_pressure(frames, vmax):
    reds = [(frame[...,0][extract_red_mask(frame)].mean()
             if extract_red_mask(frame).any() else 0) for frame in frames]
    M = max(reds) or 1
    vels = [(r / M) * vmax for r in reds]
    pressures = [np.pi * (0.25)**2 * v for v in vels]
    return vels, pressures

def bullseye_map_highlight(vals, title, cmap='jet'):
    sectors = 12
    arr = np.array(vals)
    if arr.size < sectors:
        arr = np.pad(arr, (0, sectors - arr.size), constant_values=np.nan)
    thr = np.nanmean(arr) + np.nanstd(arr)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4,4))
    width = 2 * np.pi / sectors
    for i, v in enumerate(arr):
        theta = i * width
        if np.isnan(v) or v < thr:
            color = 'white'
        else:
            norm = (v - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-6)
            color = plt.get_cmap(cmap)(norm)
        ax.bar(theta, 0.2, width=width, bottom=0.8,
               color=color, edgecolor='black', linewidth=0.8)
    ax.set_xticks(np.linspace(0, 2*np.pi, sectors, endpoint=False))
    ax.set_xticklabels([f"{i*30}°" for i in range(sectors)])
    ax.set_yticks([]); ax.set_title(title)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    return fig, arr

def get_high_sectors(arr, label):
    thr = np.nanmean(arr) + np.nanstd(arr)
    idx = np.where(arr >= thr)[0]
    if idx.size:
        degs = ", ".join(f"{i*30}°" for i in idx)
        return f"・【{label}集中】{degs} に負荷が集中しています。"
    return f"・【{label}集中】顕著な高負荷セクターはありません。"

def generate_summary_text(time, pressures, wss):
    tw = f"【WSS結果】{time[np.argmax(wss)]:.2f} 秒 に最大（{np.max(wss):.2f} Pa）を示しました。"
    tp = f"【Pressure結果】{time[np.argmax(pressures)]:.2f} 秒 に最大（{np.max(pressures):.2f}）を示しました。"
    return tw, tp

def summarize_case(wss, pressure):
    thw = np.mean(wss) + np.std(wss)
    thp = np.mean(pressure) + np.std(pressure)
    hw = np.sum(np.array(wss) > thw) / len(wss)
    hp = np.sum(np.array(pressure) > thp) / len(pressure)

    if hw == 0 or hp == 0:
        comment = "データ不足の可能性あり"
    elif hw > 0.25 and hp > 0.25:
        comment = "重度の狭窄の疑い"
    elif hw > 0.25:
        comment = "WSS極端に高い"
    elif hp > 0.25:
        comment = "Pressure極端に高い"
    elif hw > 0.15 or hp > 0.15:
        comment = "中等度の上昇傾向"
    elif hw > 0.10 or hp > 0.10:
        comment = "軽度の上昇傾向"
    else:
        comment = "異常なし"

    return round(np.max(wss), 1), round(np.max(pressure), 1), round(hw * 100, 1), round(hp * 100, 1), comment

# --- Streamlit UI ---
st.set_page_config(page_title="Vessel WSS & Pressure Analyzer", layout="wide")
st.title("Vessel Wall Pressure & Shear Stress Evaluation")

video = st.file_uploader("動画をアップロード（MP4）", type="mp4")

if video:
    st.video(video)
    vmax = st.slider("速度レンジ（cm/s）", 10.0, 120.0, 50.0)
    if st.button("解析を実行"):
        with st.spinner("解析中..."):
            frames = extract_frames(video)
            wss_maps = calculate_wss(frames)
            _, pressures = calculate_pressure(frames, vmax)
            mean_wss = np.array([np.nanmean(w) for w in wss_maps])
            time = np.arange(len(pressures)) / frame_rate

            # グラフ表示
            fig_w, axw = plt.subplots()
            axw.plot(time[:len(mean_wss)], mean_wss, color='orange')
            axw.set_title("WSS vs Time")

            fig_p, axp = plt.subplots()
            axp.plot(time, pressures, color='blue')
            axp.set_title("Pressure vs Time")

            fig_pw, axpw = plt.subplots()
            axpw.plot(time[:len(mean_wss)], pressures[:len(mean_wss)], 'b-')
            axpw2 = axpw.twinx()
            axpw2.plot(time[:len(mean_wss)], mean_wss, 'orange')
            axpw.set_title("WSS vs Pressure")

            # Bull’s Eye
            fig_be_w, arr_w = bullseye_map_highlight(mean_wss[:12], "Bull’s Eye (WSS)", cmap='Blues')
            fig_be_p, arr_p = bullseye_map_highlight(np.array(pressures[:12]), "Bull’s Eye (Pressure)", cmap='Reds')

            st.subheader("📈 計測グラフ")
            c1, c2, c3 = st.columns(3)
            c1.pyplot(fig_w); c2.pyplot(fig_p); c3.pyplot(fig_pw)

            st.subheader("🎯 Bull’s Eye Map")
            b1, b2 = st.columns(2)
            with b1:
                st.pyplot(fig_be_w)
                st.markdown(get_high_sectors(arr_w, "WSS"))
            with b2:
                st.pyplot(fig_be_p)
                st.markdown(get_high_sectors(arr_p, "Pressure"))

            # Summary
            st.markdown("### 🧠 Summary")
            with st.expander("WSSとPressureの説明"):
                st.markdown("- **WSS**：血管壁にかかるずり応力。高値は血管障害を示唆。")
                st.markdown("- **Pressure**：模擬内圧。高値は血管抵抗増加を示唆。")
            tw, tp = generate_summary_text(time, pressures, mean_wss)
            st.markdown(f"- {tw}")
            st.markdown(f"- {tp}")

            # Score
            st.markdown("### 📊 スコア結果")

            # スコア取得
            wss_max, p_max, wsr, pr, comment = summarize_case(mean_wss, pressures)
            
            # 総合判定表示 + 説明ボタン
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"- 総合判定：**{comment}**")
            with col2:
                with st.expander("🛈 説明"):
                    st.write({
                        "異常なし": "WSS・Pressure共に軽度、明らかなリスクなし。",
                        "軽度の上昇傾向": "WSS または Pressure のいずれかがやや高い。",
                        "中等度の上昇傾向": "WSS または Pressure が中程度に高い。",
                        "WSS極端に高い": "WSS が非常に高く、血管壁への負荷大。",
                        "Pressure極端に高い": "Pressure が非常に高く、内圧変化大。",
                        "重度の狭窄の疑い": "WSS・Pressure共に高く、狭窄が強く疑われます。",
                        "データ不足の可能性あり": "赤領域が検出されず、解析対象が不足している可能性があります。"
                    }.get(comment, "該当コメントなし"))
            
            # 詳細スコアはボタン展開
            if st.button("📈 詳細スコアを見る"):
                with st.expander("スコア詳細"):
                    st.markdown(f"- 最大WSS：**{wss_max} Pa**")
                    st.markdown(f"- 最大Pressure：**{p_max}**")
                    st.markdown(f"- 高WSS時間比率：**{wsr}%**")
                    st.markdown(f"- 高Pressure時間比率：**{pr}%**")

            # CSV Section
            st.markdown("### 🧾 結果CSV")
            df_time = time[:len(mean_wss)]
            df = pd.DataFrame({"Time (s)": df_time, "WSS": mean_wss, "Pressure": pressures[:len(mean_wss)]})
            st.download_button("CSV ダウンロード", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")

            # 📸 高値フレーム表示
            st.markdown("### 📸 高値フレーム表示")
            thr_w = np.nanmean(mean_wss) + np.nanstd(mean_wss)
            thr_p = np.nanmean(pressures) + np.nanstd(pressures)
            peaks_w = np.argsort(mean_wss)[-3:][::-1]
            peaks_p = np.argsort(pressures)[-3:][::-1]

            with st.expander("高WSSフレーム"):
                for i in peaks_w:
                    st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)

            with st.expander("高Pressureフレーム"):
                for i in peaks_p:
                    st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)

            with st.expander("同時に高いフレーム"):
                suspects = [i for i in range(len(mean_wss)) if mean_wss[i] > thr_w and pressures[i] > thr_p]
                if suspects:
                    for i in suspects[:3]:
                        st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)
                else:
                    st.info("該当フレームはありません。")
