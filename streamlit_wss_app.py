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
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    return (mask1 | mask2) > 0

def extract_frames(video_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_file.read()); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def calculate_wss(frames):
    gray = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (0,0),
                   fx=resize_scale, fy=resize_scale)
        for f in frames
    ]
    wss_maps = []
    for i in range(len(gray)-1):
        mask = extract_red_mask(frames[i])
        small = cv2.resize(mask.astype(np.uint8),
                           (gray[i].shape[1], gray[i].shape[0])) > 0
        flow = cv2.calcOpticalFlowFarneback(
            gray[i], gray[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        du = cv2.Sobel(flow[...,0], cv2.CV_32F, 1, 0, 3)
        dv = cv2.Sobel(flow[...,1], cv2.CV_32F, 0, 1, 3)
        wss_maps.append(np.where(small,
                                 mu * np.sqrt(du**2 + dv**2) / pixel_size_m,
                                 0))
    return wss_maps

def calculate_pressure(frames, vmax):
    reds = [
        (frame[...,0][extract_red_mask(frame)].mean()
         if extract_red_mask(frame).any() else 0)
        for frame in frames
    ]
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
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    return fig, arr

def generate_summary_text(time, pressures, wss):
    tw = f"【WSS結果】{time[np.argmax(wss)]:.2f} 秒 に最大（非表示）"
    tp = f"【Pressure結果】{time[np.argmax(pressures)]:.2f} 秒 に最大（非表示）"
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
    return None, None, round(hw * 100, 1), round(hp * 100, 1), comment

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

            # グラフ：X軸だけ残し、Y軸非表示
            fig1, ax1 = plt.subplots()
            ax1.plot(time[:len(mean_wss)], mean_wss, color='orange')
            ax1.set_xlabel("Time [s]")
            ax1.set_title("WSS の傾向")
            ax1.get_yaxis().set_visible(False)
            ax1.grid(False)

            fig2, ax2 = plt.subplots()
            ax2.plot(time[:len(mean_wss)], pressures[:len(mean_wss)], color='blue')
            ax2.set_xlabel("Time [s]")
            ax2.set_title("Pressure の傾向")
            ax2.get_yaxis().set_visible(False)
            ax2.grid(False)

            fig3, ax3 = plt.subplots()
            ax3.plot(time[:len(mean_wss)], pressures[:len(mean_wss)], color='blue')
            ax3.set_xlabel("Time [s]")
            ax3.get_yaxis().set_visible(False)
            ax4 = ax3.twinx()
            ax4.plot(time[:len(mean_wss)], mean_wss, color='orange')
            ax4.get_yaxis().set_visible(False)
            ax3.set_title("WSS と Pressure の傾向")

            st.subheader("📈 計測グラフ")
            cols = st.columns(3)
            cols[0].pyplot(fig1)
            cols[1].pyplot(fig2)
            cols[2].pyplot(fig3)

            # Bull's Eye Map（傾向可視化）
            fig_be_w, arr_w = bullseye_map_highlight(mean_wss[:12], "Bull’s Eye (WSS)", cmap='Blues')
            fig_be_p, arr_p = bullseye_map_highlight(np.array(pressures[:12]), "Bull’s Eye (Pressure)", cmap='Reds')
            st.subheader("🎯 Bull’s Eye Map")
            b1, b2 = st.columns(2)
            with b1:
                st.pyplot(fig_be_w)
                st.markdown(get_high_sectors(arr_w, "WSS"))
            with b2:
                st.pyplot(fig_be_p)
                st.markdown(get_high_sectors(arr_p, "Pressure"))

            # Summary & コメント
            st.markdown("### 🧠 Summary")
            with st.expander("WSS と Pressure の説明"):
                st.markdown("- **WSS**：数値表示なし。傾向を示すのみ。")
                st.markdown("- **Pressure**：同様に傾向を可視化し、数値は非表示。")
            tw, tp = generate_summary_text(time, pressures, mean_wss)
            st.markdown(f"- {tw}")
            st.markdown(f"- {tp}")

            st.markdown("### 📊 スコア結果")
            _, _, wsr, pr, comment = summarize_case(mean_wss, pressures)
            st.markdown(f"- 総合判定：**{comment}**")
            with st.expander("🛈 コメント説明"):
                st.write({
                    "異常なし": "WSS・Pressure共に傾向正常。",
                    "軽度の上昇傾向": "一方がやや高め。軽微な変化。",
                    "中等度の上昇傾向": f"WSS比率{wsr}%、Pressure比率{pr}% で中等度。",
                    "WSS極端に高い": "WSSの傾向が顕著。血管壁の負荷高。",
                    "Pressure極端に高い": "Pressure傾向が顕著。内圧負荷高。",
                    "重度の狭窄の疑い": f"WSS印象とPressure比率{pr}% 両方高値。",
                    "データ不足の可能性あり": "映像の赤領域検出不足。解析不能の可能性。"
                }.get(comment, ""))

            st.markdown("### 🧾 結果CSV")
            df = pd.DataFrame({"Time (s)": time[:len(mean_wss)], "WSS": mean_wss, "Pressure": pressures[:len(mean_wss)]})
            st.download_button("CSV ダウンロード", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")

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
