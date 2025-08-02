# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

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
    tmp.write(video_file.read())
    tmp.close()
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
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY),
                       (0,0), fx=resize_scale, fy=resize_scale)
            for f in frames]
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
        color = 'white' if np.isnan(v) or v < thr else plt.get_cmap(cmap)(
            (v - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-6))
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
        return f"- **{label} 集中部位**: {degs}"
    return f"- **{label} 集中部位**: なし"

def summarize_case(mean_wss, pressures):
    thr_wss = np.nanmean(mean_wss) + np.nanstd(mean_wss)
    thr_p = np.nanmean(pressures) + np.nanstd(pressures)
    hw = np.nansum(mean_wss > thr_wss) / len(mean_wss)
    hp = np.nansum(np.array(pressures) > thr_p) / len(pressures)
    if hw == 0 or hp == 0:
        comment = "データ不足"
    elif hw > 0.25 and hp > 0.25:
        comment = "重度の狭窄疑い"
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
    return round(hw*100,1), round(hp*100,1), comment

# --- Streamlit UI ---
st.set_page_config(page_title="Vessel Wall Shear Stress & Pressure Analyzer", layout="wide")
st.title("Vessel Wall Shear Stress & Pressure Analyzer")

video = st.file_uploader("動画をアップロード（MP4）", type="mp4")
vessel_diameter = st.number_input("血管径（mm）", min_value=0.1, value=5.0, step=0.1)

if video:
    st.video(video)
    vmax = st.slider("速度レンジ（cm/s）", 10.0, 120.0, 50.0, step=1.0)

    if st.button("解析を実行"):
        frames = extract_frames(video)
        wss_maps = calculate_wss(frames)
        pressures = calculate_pressure(frames, vmax)
        mean_wss = np.array([np.nanmean(w) for w in wss_maps])
        time = np.arange(len(mean_wss)) / frame_rate

        # グラフ描画
        fig_w, axw = plt.subplots()
        axw.plot(time, mean_wss, color='orange')
        axw.set_xlabel("Time")
        axw.set_title("WSS Trend")

        fig_p, axp = plt.subplots()
        axp.plot(time, pressures[:len(mean_wss)], color='blue')
        axp.set_xlabel("Time")
        axp.set_title("Pressure Trend")

        fig_pw, axpw = plt.subplots()
        axpw.plot(time, pressures[:len(mean_wss)], color='blue')
        axpw2 = axpw.twinx()
        axpw2.plot(time, mean_wss, color='orange')
        axpw.set_xlabel("Time")
        axpw.set_title("WSS & Pressure Trend")

        # 横並び
        st.subheader("📈 計測グラフ")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(fig_w)
            max_wss_frame = np.nanargmax(mean_wss)
            st.markdown(f"- 最大WSSは **{max_wss_frame/frame_rate:.2f} 秒** に確認されました。")
        with col2:
            st.pyplot(fig_p)
            max_p_frame = np.nanargmax(pressures[:len(mean_wss)])
            st.markdown(f"- 最大Pressureは **{max_p_frame/frame_rate:.2f} 秒** に確認されました。")
        with col3:
            st.pyplot(fig_pw)
            thr_wss = np.nanmean(mean_wss) + np.nanstd(mean_wss)
            thr_p = np.nanmean(pressures[:len(mean_wss)]) + np.nanstd(pressures[:len(mean_wss)])
            simul = [i for i in range(len(mean_wss)) if mean_wss[i]>thr_wss and pressures[i]>thr_p]
            if simul:
                st.markdown(f"- WSSとPressureが同時に高かったのは **{simul[0]/frame_rate:.2f} 秒** です。")
            else:
                st.markdown("- WSSとPressureが同時に高くなった瞬間は検出されませんでした。")

        # Bull’s Eye Maps
        fig_be_w, arr_w = bullseye_map_highlight(mean_wss[:12], "Bull’s Eye (WSS)", cmap='Blues')
        fig_be_p, arr_p = bullseye_map_highlight(np.array(pressures[:12]), "Bull’s Eye (Pressure)", cmap='Reds')
        st.subheader("🎯 Bull’s Eye Map")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig_be_w)
            st.markdown(get_high_sectors(arr_w, "WSS"))
        with c2:
            st.pyplot(fig_be_p)
            st.markdown(get_high_sectors(arr_p, "Pressure"))

        # Summary
        wsr, pr, comment = summarize_case(mean_wss, pressures)
        st.markdown("### 🧠 サマリー")
        st.markdown(f"- 総合判定：**{comment}**")
        with st.expander("🛈 コメント説明"):
            st.write({
                "異常なし": "全体の傾向は正常範囲です。",
                "軽度の上昇傾向": "わずかに上昇していますが傾向は軽微です。",
                "中等度の上昇傾向": f"WSS比率：{wsr}%、Pressure比率：{pr}%で中等度上昇。",
                "WSS極端に高い": "WSSの傾向が顕著に上昇しています。",
                "Pressure極端に高い": "Pressureの傾向が顕著に上昇しています。",
                "重度の狭窄疑い": "WSS・Pressure共に強く上昇、狭窄の可能性があります。",
                "データ不足": "赤色マスクが不足している可能性があります。"
            }.get(comment, ""))

        # 詳細スコア
        with st.expander("📊 詳細スコア"):
            st.markdown(f"- 高WSS時間比率：**{wsr}%**")
            st.markdown(f"- 高Pressure時間比率：**{pr}%**")

        # CSV
        # CSV出力部分
        st.markdown("### 📄 結果CSV")
        
        df = pd.DataFrame({
            "Frame": np.arange(len(mean_wss)),
            "Time (s)": time,
            "WSS": mean_wss,
            "Pressure": pressures[:len(mean_wss)]
        })
        
        st.download_button(
            label="CSVとして保存",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="results.csv",
            mime="text/csv"
        )

        # High-value Frames
        st.markdown("### 📸 高値フレーム表示")
        peaks_w = np.argsort(mean_wss)[-3:][::-1]
        peaks_p = np.argsort(pressures[:len(mean_wss)])[-3:][::-1]
        with st.expander("高WSSフレーム"):
            for i in peaks_w: st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)
        with st.expander("高Pressureフレーム"):
            for i in peaks_p: st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)
        with st.expander("同時高値フレーム"):
            suspects = [i for i in range(len(mean_wss)) if mean_wss[i]>thr_wss and pressures[i]>thr_p]
            if suspects:
                for i in suspects[:3]:
                    st.image(frames[i], caption=f"{i/frame_rate:.2f} 秒", use_column_width=True)
            else:
                st.info("該当フレームはありません。")

        st.success("解析完了！")

