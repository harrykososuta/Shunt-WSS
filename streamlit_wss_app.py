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
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (0,0),
                       fx=resize_scale, fy=resize_scale) for f in frames]
    wss_maps, centers = [], []
    for i in range(len(gray)-1):
        mask = extract_red_mask(frames[i])
        small_mask = cv2.resize(mask.astype(np.uint8),
                                (gray[i].shape[1], gray[i].shape[0])) > 0
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i+1], None,
                                            0.5,3,15,3,5,1.2,0)
        du = cv2.Sobel(flow[...,0],cv2.CV_32F,1,0,3)
        dv = cv2.Sobel(flow[...,1],cv2.CV_32F,0,1,3)
        mag = np.sqrt(du**2 + dv**2)
        wss_maps.append(np.where(small_mask, mu * mag / pixel_size_m, 0))
    return wss_maps, None

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
        vals = np.pad(vals, (0, sectors - vals.size), constant_values=np.nan)
    thr = np.nanmean(vals) + np.nanstd(vals)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4,4))
    width = 2 * np.pi / sectors
    for i, val in enumerate(vals):
        theta = i * width
        if np.isnan(val):
            color = 'lightgrey'
        elif highlight_only and val < thr:
            color = 'white'
        else:
            norm = (val - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-6)
            color = plt.get_cmap(cmap)(norm)
        ax.bar(theta, 0.2, width=width,
               bottom=0.8, color=color,
               edgecolor='black', linewidth=0.8)
    ax.set_xticks(np.linspace(0, 2*np.pi, sectors, endpoint=False))
    ax.set_xticklabels([f"{i*30}°" for i in range(sectors)])
    ax.set_yticks([]); ax.set_title(title)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    return fig, vals

def get_high_sectors(vals, label):
    thr = np.nanmean(vals) + np.nanstd(vals)
    indices = np.where(vals >= thr)[0]
    if indices.size:
        degs = [f"{i*30}°" for i in indices]
        return f"・【{label}集中】{', '.join(degs)} に負荷が集中しています。"
    else:
        return f"・【{label}集中】明確な高負荷セクターはありません。"

def generate_summary_text(time, pressures, wss):
    tw = f"【WSS結果】{time[np.argmax(wss)]:.2f} 秒 に最大（{np.max(wss):.2f} Pa）を示しました。"
    tp = f"【Pressure結果】{time[np.argmax(pressures)]:.2f} 秒 に最大（{np.max(pressures):.2f}）を示しました。"
    return tw, tp

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
    return round(np.max(wss),1), round(np.max(pressure),1), round(hw*100,1), round(hp*100,1), comment

# --- Streamlit UI ---
st.set_page_config(page_title="Vessel WSS & Pressure Analyzer", layout="wide")
st.title("Vessel Wall Pressure ＆ Shear Stress Evaluation")

video = st.file_uploader("動画をアップロード（MP4）", type="mp4")
if video:
    st.video(video)
    vmax = st.slider("速度レンジ（cm/s）", 10.0, 120.0, 50.0)
    if st.button("解析を実行"):
        with st.spinner("解析中..."):
            frames = extract_frames(video)
            wss_maps, _ = calculate_wss(frames)
            _, pressures = calculate_pressure(frames, vmax)
            mean_wss = np.array([np.nanmean(w) for w in wss_maps])
            time = np.arange(len(pressures)) / frame_rate

            # グラフ
            fig_w, axw = plt.subplots(); axw.plot(time[:len(mean_wss)], mean_wss, color='orange'); axw.set_title("WSS vs Time")
            fig_p, axp = plt.subplots(); axp.plot(time, pressures, color='blue'); axp.set_title("Pressure vs Time")
            fig_pw, axpw = plt.subplots(); axpw.plot(time[:len(mean_wss)], pressures[:len(mean_wss)], 'b-'); axpw2=axpw.twinx(); axpw2.plot(time[:len(mean_wss)], mean_wss, 'orange'); axpw.set_title("WSS vs Pressure")

            # Bull’s Eye + 集中角度
            fig_be_w, vals_w = bullseye_map_highlight(mean_wss[:12], "Bull’s Eye (WSS)", cmap='Blues')
            fig_be_p, vals_p = bullseye_map_highlight(np.array(pressures[:12]), "Bull’s Eye (Pressure)", cmap='Reds')

            st.subheader("計測グラフ")
            c1, c2, c3 = st.columns(3)
            c1.pyplot(fig_w); c2.pyplot(fig_p); c3.pyplot(fig_pw)

            st.subheader("Bull’s Eye Map")
            b1, b2 = st.columns(2)
            with b1:
                st.pyplot(fig_be_w)
                st.markdown(get_high_sectors(vals_w, "WSS"))
            with b2:
                st.pyplot(fig_be_p)
                st.markdown(get_high_sectors(vals_p, "Pressure"))

            # Summary
            st.markdown("<div style='background:#fff; padding:20px; border-radius:10px; border:1px solid #ccc;'>", unsafe_allow_html=True)
            with st.expander("WSSとPressureの説明"):
                st.markdown("""
- **WSS**：血管壁にかかるずり応力。高いと血管障害の兆候。
- **Pressure**：模擬的な血管内圧。高いと血流抵抗・壁応力を示唆。
""")
            tw, tp = generate_summary_text(time, pressures, mean_wss)
            st.markdown(f"- {tw}")
            st.markdown(f"- {tp}")
            st.markdown("</div>", unsafe_allow_html=True)

            # スコア
            wss_max, p_max, wsr, pr, comment = summarize_case(mean_wss, pressures)
            severity = "#d1e7dd" if comment == "異常なし" else "#fff3cd" if "可能性" in comment else "#f8d7da"
            st.markdown(f"<div style='background:#fff; padding:20px; border-radius:10px; border:1px solid #ccc;'>", unsafe_allow_html=True)
            st.markdown("**📊 スコア結果**")
            st.markdown(f"・最大WSS：**{wss_max} Pa**  \n・最大Pressure：**{p_max}**  \n・高WSS時間比率：**{wsr}%**  \n・高Pressure時間比率：**{pr}%**")
            st.markdown(f"<div style='background:{severity}; padding:15px; border-radius:10px; text-align:center; font-weight:bold;'>総合判定：{comment}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # CSV
            st.markdown(f"<div style='background:#fff; padding:20px; border-radius:10px; border:1px solid #ccc;'>", unsafe_allow_html=True)
            st.markdown("**📋 結果CSV**")
            df = pd.DataFrame([{"WSS最大 [Pa]":wss_max, "Pressure最大":p_max, "高WSS時間比率 [%]":wsr, "高Pressure時間比率 [%]":pr, "コメント":comment}])
            st.download_button("CSVとして保存", data=df.to_csv(index=False).encode(), file_name="summary.csv")
            st.markdown("</div>", unsafe_allow_html=True)

            # 高値フレーム
            st.markdown(f"<div style='background:#fff; padding:20px; border-radius:10px; border:1px solid #ccc;'>", unsafe_allow_html=True)
            st.markdown("**📸 高値フレーム表示**")
            thr_w = np.nanmean(mean_wss) + np.nanstd(mean_wss)
            thr_p = np.nanmean(pressures) + np.nanstd(pressures)
            peaks_w = np.argsort(mean_wss)[-3:][::-1]
            peaks_p = np.argsort(pressures)[-3:][::-1]
            with st.expander("高WSSフレーム"):
                for i in peaks_w: st.image(frames[i], caption=f"Frame {i} – {i/frame_rate:.2f}s")
            with st.expander("高Pressureフレーム"):
                for i in peaks_p: st.image(frames[i], caption=f"Frame {i} – {i/frame_rate:.2f}s")
            with st.expander("WSSとPressureが同時に高いフレーム"):
                suspects = [i for i in range(len(mean_wss)) if mean_wss[i] > thr_w and pressures[i] > thr_p]
                if suspects:
                    for i in suspects[:3]: st.image(frames[i], caption=f"Frame {i} – {i/frame_rate:.2f}s")
                else:
                    st.info("該当フレームなし。")
            st.markdown("</div>", unsafe_allow_html=True)

            st.success("解析完了！")
