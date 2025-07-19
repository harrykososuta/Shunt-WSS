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

# --- ユーティリティ関数 ---
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

def calculate_pressure(frames, vmax):
    red_means = [(frame[...,0][extract_red_mask(frame)].mean() if extract_red_mask(frame).any() else 0) for frame in frames]
    M = max(red_means) or 1
    return [(m/M)*vmax for m in red_means], [np.pi*(0.25)**2*(m/M)*vmax for m in red_means]

def bullseye_map(values, title):
    rings, sectors = 3, 12
    vals = np.array(values)
    if vals.size < rings*sectors:
        vals = np.pad(vals, (0, rings*sectors - vals.size), 'constant', constant_values=np.nan)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4,4))
    width = 2*np.pi / sectors
    for r in range(rings):
        for s in range(sectors):
            idx = r*sectors + s
            ax.bar(s*width, (r+1)/rings - r/rings, width=width, bottom=r/rings,
                   color=plt.cm.jet(np.nan_to_num(vals[idx])), edgecolor='white')
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels([f"{i*30}°" for i in range(12)])
    ax.set_yticks([])
    ax.set_title(title)
    return fig, vals

def generate_summary_text(time, pressures, wss):
    t_p = time[np.argmax(pressures)]
    t_w = time[np.argmax(wss)]
    text_p = f"Pressureは約 **{t_p:.2f}s** に最大（{pressures.max():.2f}）を示し、高負荷が観察されました。"
    text_w = f"WSSは約 **{t_w:.2f}s** に最大（{wss.max():.2f}）を示し、内皮細胞へのずり応力がピークでした。"
    return text_p, text_w

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
st.set_page_config(page_title="Analyzer", layout="wide")
st.title("🧐 Vessel Wall Pressure & Shear Stress Evaluation")

video_file = st.file_uploader("Upload MP4", type="mp4")
if video_file:
    st.video(video_file)
    vmax = st.slider("速度レンジ (cm/s)", 10.0, 120.0, 50.0)
    if st.button("解析を実行"):
        with st.spinner("解析中..."):
            frames = extract_frames(video_file)
            wss_maps, centers = calculate_wss(frames)
            vels, pressures = calculate_pressure(frames, vmax)
            mean_wss = np.array([np.nanmean(w) for w in wss_maps])
            time = np.arange(len(pressures)) / frame_rate

            # 各グラフ生成
            fig_w, axw = plt.subplots(); axw.plot(time[:len(mean_wss)], mean_wss, color='orange'); axw.set_title("WSS vs Time"); axw.set_xlabel("Time [s]")
            fig_p, axp = plt.subplots(); axp.plot(time, pressures, color='blue'); axp.set_title("Pressure vs Time"); axp.set_xlabel("Time [s]")
            fig_pw, axpw = plt.subplots(); axpw.plot(time[:len(mean_wss)], pressures[:len(mean_wss)], 'b-'); axpw2=axpw.twinx(); axpw2.plot(time[:len(mean_wss)], mean_wss, 'orange'); axpw.set_title("WSS vs Pressure"); axpw.set_xlabel("Time [s]")

            # Bull's Eye
            fig_be_w, vals_w = bullseye_map(mean_wss, "Bull's Eye (WSS)")
            fig_be_p, vals_p = bullseye_map(np.array(pressures)[:len(mean_wss)], "Bull's Eye (Pressure)")

            # レイアウト
            cols = st.columns(3)
            cols[0].pyplot(fig_w); cols[1].pyplot(fig_p); cols[2].pyplot(fig_pw)
            be_cols = st.columns(2)
            be_cols[0].pyplot(fig_be_w); be_cols[1].pyplot(fig_be_p)

            st.markdown("---")
            # Summary 表示
            txt_p, txt_w = generate_summary_text(time, np.array(pressures), mean_wss)
            st.write(txt_w); st.write(txt_p)

            wss_max, p_max, wsr, pr, comment = summarize_case(mean_wss, pressures)
            st.markdown(f"**スコア結果：** WSS最大={wss_max} Pa, Pressure最大={p_max}, 高WSS時間比率={wsr}%, 高Pressure時間比率={pr}%, 判定：{comment}")

            st.markdown("---")
            # CSVと画像
            df = pd.DataFrame([{"WSS最大":wss_max,"Pressure最大":p_max,"高WSS率":wsr,"高Pressure率":pr,"コメント":comment}])
            st.download_button("CSVとして保存", data=df.to_csv(index=False).encode(), file_name="summary.csv")
            thr_w, thr_p = np.mean(mean_wss)+np.std(mean_wss), np.mean(pressures)+np.std(pressures)
            peaks_w = np.argsort(mean_wss)[-3:][::-1]
            peaks_p = np.argsort(pressures)[-3:][::-1]

            with st.expander("📸 高WSSが観察されたフレーム"):
                for i in peaks_w: st.image(frames[i], caption=f"{i} - {i/frame_rate:.2f}s")
            with st.expander("📸 高Pressureが観察されたフレーム"):
                for i in peaks_p: st.image(frames[i], caption=f"{i} - {i/frame_rate:.2f}s")
            with st.expander("⚠️ 同時高WSSとPressure"):
                suspect = [i for i in range(len(mean_wss)) if mean_wss[i]>thr_w and pressures[i]>thr_p]
                if suspect:
                    for i in suspect[:3]: st.image(frames[i], caption=f"{i} - {i/frame_rate:.2f}s")
                else:
                    st.info("該当フレームなし")

            st.success("解析完了！")
