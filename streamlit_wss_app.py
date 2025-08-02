# streamlit_app.py
import streamlit as st
import cv2, numpy as np, math, pandas as pd
import matplotlib.pyplot as plt
import tempfile
from scipy.signal import correlate

# --- パラメータ ---
mu = 0.0035
pixel_size_m = 1e-4
resize_scale = 0.5
frame_rate = 30.0

# --- ユーティリティ関数 ---
def extract_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return ((cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255])) |
             cv2.inRange(hsv, np.array([160,70,50]), np.array([180,255,255])))) > 0

def extract_frames(video_file):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
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
        du = cv2.Sobel(flow[...,0], cv2.CV_32F, 1, 0, 3)
        dv = cv2.Sobel(flow[...,1], cv2.CV_32F, 0, 1, 3)
        wss_maps.append(np.where(small,
                                 mu * np.sqrt(du**2 + dv**2) / pixel_size_m,
                                 np.nan))
    return wss_maps

def calculate_pressure(frames, vmax):
    reds = [(frame[...,0][extract_red_mask(frame)].mean()
             if extract_red_mask(frame).any() else np.nan)
            for frame in frames]
    M = max([r for r in reds if not np.isnan(r)], default=1)
    pressures = [(r / M) * vmax * np.pi * (0.25**2) if not np.isnan(r) else np.nan
                 for r in reds]
    return pressures

def detect_local_peaks(series):
    data = np.array(series)
    peaks = []
    for i in range(1, len(data)-1):
        if not math.isnan(data[i]) and data[i]>=data[i-1] and data[i]>=data[i+1]:
            peaks.append(i)
    return peaks

def compute_feature_from_trends(pressure, mean_wss, time):
    valid = ~np.isnan(pressure) & ~np.isnan(mean_wss)
    p,w,t = pressure[valid], mean_wss[valid], time[valid]
    if len(p) < 3:
        return {'corr_pressure_wss': np.nan,
                'lag_sec_wss_after_pressure': np.nan,
                'simultaneous_peak_counts': 0}
    corr = np.corrcoef(p, w)[0,1]
    cc = correlate(p - np.mean(p), w - np.mean(w), mode='full')
    lag = (np.argmax(cc) - (len(p)-1)) * (t[1] - t[0] if len(t)>1 else 0)
    sim = sum(any(abs(pw-pp)<=1 for pp in detect_local_peaks(p))
              for pw in detect_local_peaks(w))
    return {'corr_pressure_wss': corr,
            'lag_sec_wss_after_pressure': lag,
            'simultaneous_peak_counts': sim}

def classify_stenosis(feat, ref_stats=None):
    sim, lag = feat['simultaneous_peak_counts'], feat['lag_sec_wss_after_pressure']
    corr = feat.get('corr_pressure_wss', None)
    if sim <= 62.5:
        if lag <= 3.28:
            if corr is not None and corr <= 0.08:
                category, rule = "高度狭窄疑い", "sim<=62.5 & lag<=3.28 & corr<=0.08"
            else:
                category, rule = "狭窄なし", "sim<=62.5 & lag<=3.28 & corr>0.08"
        else:
            category, rule = "中等度狭窄疑い", "sim<=62.5 & lag>3.28"
    else:
        if lag <= 2.28:
            category, rule = "軽度狭窄疑い", "sim>62.5 & lag<=2.28"
        elif lag <= 8.62:
            category, rule = "高度狭窄疑い", "sim>62.5 & 2.28<lag<=8.62"
        else:
            category, rule = "軽度狭窄疑い", "sim>62.5 & lag>8.62 exception"
    mild_score = None
    if ref_stats:
        z = lambda x,m,s: (x-m)/s if s and s>0 else 0.0
        mild_score = z(sim, ref_stats['sim_peak_mean'], ref_stats['sim_peak_std']) + \
                      z(lag, ref_stats['lag_mean'], ref_stats['lag_std'])
        if category == "狭窄なし" and mild_score > 0.5:
            category = "軽度狭窄疑い（補正）"
            rule += f"; mild_score={mild_score:.2f}>0.5補正"
    return {'category': category, 'rule_used': rule, 'mild_suspicion_score': mild_score}

def bullseye_map_highlight(vals, title, cmap='jet'):
    sectors = 12; arr = np.array(vals)
    if arr.size < sectors: arr = np.pad(arr, (0, sectors-arr.size), constant_values=np.nan)
    thr = np.nanmean(arr) + np.nanstd(arr)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4,4))
    width = 2*np.pi/sectors
    for i, v in enumerate(arr):
        theta = i*width
        color = 'white' if np.isnan(v) or v<thr else plt.get_cmap(cmap)(
            (v-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr)+1e-6))
        ax.bar(theta, 0.2, width=width, bottom=0.8, color=color, edgecolor='black', linewidth=0.8)
    ax.set_xticks(np.linspace(0,2*np.pi,sectors,endpoint=False))
    ax.set_xticklabels([f"{i*30}°" for i in range(sectors)])
    ax.set_yticks([]); ax.set_title(title)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    return fig, arr

def get_high_sectors(arr, label):
    thr = np.nanmean(arr) + np.nanstd(arr)
    idx = np.where(arr >= thr)[0]
    if idx.size:
        degs = ", ".join(f"{i*30}°" for i in idx)
        return f"- **{label} 集中部位**: {degs}"
    return f"- **{label} 集中部位**: なし"

# --- Streamlit UI ---
st.set_page_config(page_title="Vessel Analyzer", layout="wide")
st.title("Vessel Wall Shear Stress & Pressure Analyzer")

video = st.file_uploader("動画をアップロード（MP4）", type="mp4")
vessel_diameter = st.number_input("血管径（mm）", min_value=0.1, max_value=20.0, value=5.0, step=0.1)

if video:
    st.video(video)
    vmax = st.slider("速度レンジ (cm/s)", 10.0, 120.0, 50.0, step=1.0)
    if st.button("解析を実行"):
        frames = extract_frames(video)
        wss_maps = calculate_wss(frames)
        pressures = calculate_pressure(frames, vmax)
        mean_wss = np.array([np.nanmean(w) for w in wss_maps])
        time = np.arange(len(mean_wss)) / frame_rate

        feat = compute_feature_from_trends(np.array(pressures[:len(mean_wss)]), mean_wss, time)
        ref = {"sim_peak_mean":50.0,"sim_peak_std":15.0,"lag_mean":1.5,"lag_std":1.0}
        cls = classify_stenosis(feat, ref)

        # プロット
        fig_w, axw = plt.subplots(); axw.plot(time, mean_wss, color='orange'); axw.set_xlabel("Time"); axw.set_title("WSS Trend")
        fig_p, axp = plt.subplots(); axp.plot(time, pressures[:len(mean_wss)], color='blue'); axp.set_xlabel("Time"); axp.set_title("Pressure Trend")
        fig_pw, axpw = plt.subplots(); axpw.plot(time, pressures[:len(mean_wss)], color='blue'); axpw2 = axpw.twinx(); axpw2.plot(time, mean_wss, color='orange'); axpw.set_xlabel("Time"); axpw.set_title("WSS & Pressure Trend")

        st.subheader("📈 計測グラフ")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.pyplot(fig_w)
            st.markdown(f"- 最大WSS時: **{np.nanargmax(mean_wss)/frame_rate:.2f} 秒**")
        with c2:
            st.pyplot(fig_p)
            st.markdown(f"- 最大Pressure時: **{np.nanargmax(pressures[:len(mean_wss)])/frame_rate:.2f} 秒**")
        with c3:
            st.pyplot(fig_pw)
            st.markdown(f"- 同時ピーク分類: **{cls['category']}**")
            st.markdown(f"  ルール: {cls['rule_used']}")

        # Bull’s Eye
        fig_be_w, arr_w = bullseye_map_highlight(mean_wss[:12], "Bull’s Eye (WSS)", "Blues")
        fig_be_p, arr_p = bullseye_map_highlight(np.array(pressures[:12]), "Bull’s Eye (Pressure)", "Reds")
        st.subheader("🎯 Bull’s Eye Map")
        b1,b2 = st.columns(2)
        with b1:
            st.pyplot(fig_be_w)
            st.markdown(get_high_sectors(arr_w, "WSS"))
        with b2:
            st.pyplot(fig_be_p)
            st.markdown(get_high_sectors(arr_p, "Pressure"))

        # 判定結果
        st.markdown("### 🧠 判定結果")
        st.markdown(f"- カテゴリ: **{cls['category']}**")
        if cls.get('mild_suspicion_score') is not None:
            st.markdown(f"- Mild suspicion score: **{cls['mild_suspicion_score']:.2f}**")

        with st.expander("🔍 特徴量詳細"):
            st.write({
                "Correlation (Pressure vs WSS)": f"{feat['corr_pressure_wss']:.2f} — 0 に近いほど無関係、±1 に近いほど強い線形関係です。",
                "Lag (WSS lag after Pressure) [s]": f"{feat['lag_sec_wss_after_pressure']:.2f} — 正の値なら WSS が Pressure より後にピークが来ています。",
                "Simultaneous Peak Count": f"{feat['simultaneous_peak_counts']} — WSS と Pressure のピークが同時に発生した回数（近接）です。"
            })

        # CSV 出力
        st.markdown("### 📄 結果CSV")
        df = pd.DataFrame({
            "Frame": np.arange(len(mean_wss)),
            "Time_s": time,
            "WSS": mean_wss,
            "Pressure": pressures[:len(mean_wss)],
            "Category": cls['category'],
            "Rule": cls['rule_used']
        })
        st.download_button("CSVとして保存", df.to_csv(index=False).encode("utf-8"), file_name="results.csv", mime="text/csv")

        # 高値フレーム
        st.markdown("### 📸 高値フレーム表示")
        thr_wss = np.nanmean(mean_wss) + np.nanstd(mean_wss)
        thr_p = np.nanmean(pressures[:len(mean_wss)]) + np.nanstd(pressures[:len(mean_wss)])
        peaks_w = np.argsort(mean_wss)[-3:][::-1]
        peaks_p = np.argsort(pressures[:len(mean_wss)])[-3:][::-1]
        with st.expander("高WSSフレーム"):
            for i in peaks_w:
                st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)
        with st.expander("高Pressureフレーム"):
            for i in peaks_p:
                st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)
        with st.expander("同時高値フレーム"):
            suspects = [i for i in range(len(mean_wss)) if mean_wss[i] > thr_wss and pressures[i] > thr_p]
            if suspects:
                for i in suspects[:3]:
                    st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)
            else:
                st.info("該当フレームなし")

        st.success("解析完了！")

