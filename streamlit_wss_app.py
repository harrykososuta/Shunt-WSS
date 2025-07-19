# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from PIL import Image

# --- Parameters ---
mu = 0.0035
pixel_size_m = 1e-4
resize_scale = 0.5
frame_rate = 30.0

# --- Utility Functions ---
def extract_red_mask(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower1, upper1 = np.array([0,70,50]), np.array([10,255,255])
    lower2, upper2 = np.array([160,70,50]), np.array([180,255,255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return (m1 | m2) > 0

def extract_frames(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes.read())
        tmp = f.name
    cap = cv2.VideoCapture(tmp)
    frames=[]
    while True:
        ret,frame=cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def calculate_wss(frames):
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY),(0,0),fx=resize_scale,fy=resize_scale)
            for f in frames]
    wss_maps, centers = [], []
    for i in range(len(gray)-1):
        mask=extract_red_mask(frames[i])
        small_mask=cv2.resize(mask.astype(np.uint8),(gray[i].shape[1],gray[i].shape[0]))>0
        coords=np.column_stack(np.where(mask))
        cy,cx=np.mean(coords,axis=0).astype(int)
        centers.append((int(cx*resize_scale), int(cy*resize_scale)))
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i+1], None,
                                            0.5,3,15,3,5,1.2,0)
        du = cv2.Sobel(flow[...,0],cv2.CV_32F,1,0,ksize=3)
        dv = cv2.Sobel(flow[...,1],cv2.CV_32F,0,1,ksize=3)
        gm = np.sqrt(du**2 + dv**2)
        wss = mu*gm/pixel_size_m
        wss_maps.append(np.where(small_mask, wss, 0))
    return wss_maps, centers

def calculate_pressure(frames, vmax):
    means=[]
    for f in frames:
        mask=extract_red_mask(f)
        if mask.any():
            means.append(np.mean(f[...,0][mask]))
        else:
            means.append(0)
    M=max(means) or 1
    vels=[(m/M)*vmax for m in means]
    A=np.pi*(0.25)**2
    return vels, [A*v for v in vels]

def bullseye_map(values, title):
    rings, sectors = 3, 6
    values = values if len(values)==rings*sectors else np.random.rand(rings*sectors)
    fig,ax=plt.subplots(figsize=(4,4),subplot_kw=dict(polar=True))
    width = 2*np.pi/sectors
    for r in range(rings):
        for s in range(sectors):
            idx=r*sectors+s
            ax.bar(s*width, (r+1)/rings - r/rings, width=width,
                   bottom=r/rings, color=plt.cm.jet(values[idx]), edgecolor='white')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    return fig, values

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
    high_wss_ratio = np.sum(np.array(wss) > high_wss_threshold) / len(wss)
    high_pressure_ratio = np.sum(np.array(pressure) > high_pressure_threshold) / len(pressure)
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
st.set_page_config(page_title="Analyzer", layout="wide")
st.title("🧐 Vessel Wall Pressure & Shear Stress Evaluation")

video = st.file_uploader("Upload MP4", type="mp4")
if video:
    velocity_range = st.slider("速度レンジ (cm/s)", 10.0,120.0,50.0)
    if st.button("解析を実行"):
        with st.spinner("解析中..."):
            frames = extract_frames(video)
            wss_maps, centers = calculate_wss(frames)
            vels, pressures = calculate_pressure(frames, velocity_range)
            mean_wss = [np.mean(w[w>0]) for w in wss_maps]
            time = np.arange(len(pressures))/frame_rate

            fig_p, ax_p = plt.subplots(); ax_p.plot(time, pressures, color='blue'); ax_p.set_title("Pressure vs Time")
            fig_w, ax_w = plt.subplots(); ax_w.plot(time[:len(mean_wss)], mean_wss, color='orange'); ax_w.set_title("WSS vs Time")
            fig_pw, ax_pw = plt.subplots(); ax_pw.plot(time[:len(mean_wss)], pressures[:len(mean_wss)], color='blue'); ax_pw2 = ax_pw.twinx(); ax_pw2.plot(time[:len(mean_wss)], mean_wss, color='orange'); fig_pw.tight_layout(); ax_pw.set_title("WSS vs Pressure")

            fig_be_w, m_w = bullseye_map(mean_wss[:18], "Bull's Eye (WSS)")
            fig_be_p, m_p = bullseye_map(pressures[:18], "Bull's Eye (Pressure)")

            st.columns(3)[0].pyplot(fig_w)
            st.columns(3)[1].pyplot(fig_p)
            st.columns(3)[2].pyplot(fig_pw)
            st.columns(2)[0].pyplot(fig_be_w)
            st.columns(2)[1].pyplot(fig_be_p)

            st.markdown("---")
            st.subheader("🧠 Summary")
            st.markdown("<div style='background:white;padding:10px;border-radius:8px;'>", unsafe_allow_html=True)
            wss_max, p_max, wss_ratio, p_ratio, comment = summarize_case(mean_wss, pressures)
            st.info(generate_summary(pressures, mean_wss))
            st.write({
                "WSS最大 [Pa]": wss_max,
                "Pressure最大": p_max,
                "高WSS時間比率 [%]": wss_ratio,
                "高Pressure時間比率 [%]": p_ratio,
                "コメント": comment
            })
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📋 結果のCSV出力")
            st.markdown("<div style='background:white;padding:10px;border-radius:8px;'>", unsafe_allow_html=True)
            df = pd.DataFrame([{
                "WSS最大 [Pa]": wss_max,
                "Pressure最大": p_max,
                "高WSS時間比率 [%]": wss_ratio,
                "高Pressure時間比率 [%]": p_ratio,
                "コメント": comment
            }])
            st.download_button("CSVとして保存", data=df.to_csv(index=False).encode("utf-8"), file_name="summary.csv")

            threshold_w = np.mean(mean_wss) + np.std(mean_wss)
            threshold_p = np.mean(pressures) + np.std(pressures)
            peaks_w = sorted(range(len(mean_wss)), key=lambda i: mean_wss[i], reverse=True)[:3]
            peaks_p = sorted(range(len(pressures)), key=lambda i: pressures[i], reverse=True)[:3]

            with st.expander("📸 高WSSが観察されたフレーム"):
                for idx in peaks_w:
                    st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)

            with st.expander("📸 高Pressureが観察されたフレーム"):
                for idx in peaks_p:
                    st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)

            suspect_frames = [i for i in range(len(mean_wss)) if mean_wss[i]>threshold_w and pressures[i]>threshold_p]
            with st.expander("⚠️ WSSとPressureが同時に高かったフレーム（狭窄の可能性）"):
                if suspect_frames:
                    for idx in suspect_frames[:3]:
                        st.image(frames[idx], caption=f"Frame {idx} – {idx/frame_rate:.2f}s", use_column_width=True)
                else:
                    st.info("⚠️ 同時に高値を示すフレームはありません。")
            st.markdown("</div>", unsafe_allow_html=True)

            st.success("解析完了！")
