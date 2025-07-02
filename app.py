import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import whisper
import tempfile
import soundfile as sf
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="－音声分析アプリ－", layout="wide")
st.markdown("# 🗣️ 音声分析用のWEBアプリ") 

# ===== 共通関数 =====

def convert_to_wav(uploaded_file):
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    return tmp_path

def analyze_features(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    clarity = librosa.feature.spectral_flatness(y=y)[0]
    return {
        "duration": duration,
        "rms": rms,
        "pitch": pitch,
        "clarity": clarity,
        "rms_mean": np.mean(rms),
        "pitch_mean": np.mean(pitch),
        "pitch_std": np.std(pitch),
        "clarity_mean": np.mean(clarity),
    }

def generate_feedback(feat):
    fb = []
    if feat["rms_mean"] < 0.01:
        fb.append("声量が控えめです。")
    elif feat["rms_mean"] > 0.03:
        fb.append("十分な声量があり、力強い発話です。")
    else:
        fb.append("適度な音量で話せています。")

    if feat["pitch_mean"] < 110:
        fb.append("やや低めの声です。")
    elif feat["pitch_mean"] > 250:
        fb.append("比較的高めの声です。")
    else:
        fb.append("安定した音程です。")

    if feat["clarity_mean"] < 0.2:
        fb.append("発話が明瞭で聞き取りやすいです。")
    elif feat["clarity_mean"] > 0.4:
        fb.append("ややこもった音に聞こえる可能性があります。")

    return " ".join(fb)

# ================= 録音＆Whisper解析 =================

st.header("🎙️ 録音＆文字起こし")
st.markdown("##### ↓↓を押して開始・停止を操作します")

wav_audio = audio_recorder(pause_threshold=8.0, sample_rate=16000)

if wav_audio is None:
    st.info("🟢 マイクが黒で待機中…赤で録音中です")
else:
    st.success("🔴 録音完了！再生・保存・分析できます")

if wav_audio:
    st.audio(wav_audio, format="audio/wav")
    st.download_button("⬇️ ここから録音を保存できます", wav_audio, file_name="recorded.wav")

    if st.button("📊 録音音声を音響的に分析する"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(wav_audio)
            tmp_audio.flush()
            y_rec, sr_rec = librosa.load(tmp_audio.name, sr=None)

        feat_rec = analyze_features(y_rec, sr_rec)

        st.subheader("📊 録音音声の音響指標")
        col1, col2, col3 = st.columns(3)
        col1.metric("⏱ 長さ", f"{feat_rec['duration']:.2f}s")
        col2.metric("🔊 平均音量", f"{feat_rec['rms_mean']:.4f}")
        col3.metric("🎵 平均ピッチ", f"{feat_rec['pitch_mean']:.2f}Hz")
        st.metric("🗣 明瞭度", f"{feat_rec['clarity_mean']:.4f}")
        st.info(generate_feedback(feat_rec))

        st.subheader("📈 ピッチ・音量グラフ（録音音声）")
        fig_rec, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax[0].plot(librosa.times_like(feat_rec["rms"], sr=sr_rec), feat_rec["rms"], color="blue")
        ax[1].plot(librosa.times_like(feat_rec["pitch"], sr=sr_rec), feat_rec["pitch"], color="green")
        ax[1].set_xlabel("Time(s)")
        st.pyplot(fig_rec)

    if st.button("🔍 Whisper文字起こしを実行する"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_audio)
            tmp.flush()
            y, sr = librosa.load(tmp.name, sr=None)

        with st.spinner("Whisperで文字起こし中..."):
            model = whisper.load_model("small")
            result = model.transcribe(tmp.name, language="ja")

        st.subheader("📝 Whisper文字起こし")
        st.write(result["text"])

        st.subheader("📋 セグメント評価")
        seg_data = []
        for seg in result["segments"]:
            s_start = seg["start"]
            s_end = seg["end"]
            dur = s_end - s_start
            text = seg["text"].strip()
            words = len(text.replace("　", " ").split())
            rate = round(words / dur, 2) if dur > 0 else 0

            y_part = y[int(s_start * sr):int(s_end * sr)]
            f0 = librosa.yin(y_part, fmin=50, fmax=500, sr=sr)
            f0_mean = round(np.mean(f0), 1)
            f0_std = round(np.std(f0), 1)
            rms = librosa.feature.rms(y=y_part)[0]
            rms_mean = round(np.mean(rms), 4)
            energy = rms < 0.01
            pause_ratio = round(np.sum(energy) / len(rms) * 100, 1)

            comments = []
            if rate < 2.5:
                comments.append("ややゆっくり")
            elif rate > 5:
                comments.append("速いペース")
            if f0_std < 15:
                comments.append("単調な抑揺")
            elif f0_std > 30:
                comments.append("抑揺が豊か")
            if pause_ratio > 30:
                comments.append("ポーズ多め")
            if not comments:
                comments.append("比較的安定")

            seg_data.append({
                "区間": f"{s_start:.1f}-{s_end:.1f}s",
                "語数": words,
                "話速": f"{rate}語/秒",
                "F0平均": f"{f0_mean}Hz",
                "F0ばらつき": f"{f0_std}Hz",
                "無音率": f"{pause_ratio}%",
                "音量": rms_mean,
                "内容": text,
                "コメント": "、".join(comments)
            })

        st.dataframe(seg_data, use_container_width=True)

# ========== 単体音声ファイルの分析 ==========
st.header("📂 音声データを分析")

uploaded_file = st.file_uploader("音声ファイルをアップロード（WAV推奨）", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file)
    path = convert_to_wav(uploaded_file)
    y, sr = librosa.load(path, sr=None)
    feat = analyze_features(y, sr)

    st.subheader("📊 音響指標")
    col1, col2, col3 = st.columns(3)
    col1.metric("⏱ 長さ", f"{feat['duration']:.2f}s")
    col2.metric("🔊 平均音量", f"{feat['rms_mean']:.4f}")
    col3.metric("🎵 平均ピッチ", f"{feat['pitch_mean']:.2f}Hz")
    st.metric("🗣 明瞭度", f"{feat['clarity_mean']:.4f}")
    st.info(generate_feedback(feat))

    st.subheader("📈 ピッチ・音量グラフ")
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(librosa.times_like(feat["rms"], sr=sr), feat["rms"], color="blue")
    ax[1].plot(librosa.times_like(feat["pitch"], sr=sr), feat["pitch"], color="green")
    ax[1].set_xlabel("時間（秒）")
    st.pyplot(fig)

# ========== 音声A/B 比較分析 ==========
st.header("📂 音声の比較分析")

def compare_features(fa, fb):
    def pct(a, b):
        return f"{(b - a) / a * 100:+.1f}%" if a != 0 else "N/A"
    return {
        "音声長": f"{fa['duration']:.2f}s → {fb['duration']:.2f}s",
        "平均音量": f"{fa['rms_mean']:.4f} → {fb['rms_mean']:.4f}（{pct(fa['rms_mean'], fb['rms_mean'])}）",
        "平均ピッチ": f"{fa['pitch_mean']:.2f}Hz → {fb['pitch_mean']:.2f}Hz（{pct(fa['pitch_mean'], fb['pitch_mean'])}）",
        "ピッチばらつき": f"{fa['pitch_std']:.2f}Hz → {fb['pitch_std']:.2f}Hz（{pct(fa['pitch_std'], fb['pitch_std'])}）",
        "明瞭度": f"{fa['clarity_mean']:.4f} → {fb['clarity_mean']:.4f}（{pct(fa['clarity_mean'], fb['clarity_mean'])}）",
    }

def compare_feedback(fa, fb):
    msg = []
    if fb["rms_mean"] > fa["rms_mean"]:
        msg.append("Bの方が声量が大きいです。")
    else:
        msg.append("Aの方が声量が大きいです。")
    if fb["pitch_std"] < fa["pitch_std"]:
        msg.append("Bは音程が安定しています。")
    else:
        msg.append("Aの方が抑揺があります。")
    if fb["clarity_mean"] < fa["clarity_mean"]:
        msg.append("Bは明瞭度が高めです。")
    else:
        msg.append("Aは明瞭度が高めです。")
    return " ".join(msg)

col1, col2 = st.columns(2)
file_a = col1.file_uploader("音声A", type=["wav"], key="compare_a")
file_b = col2.file_uploader("音声B", type=["wav"], key="compare_b")

if file_a and file_b:
    st.audio(file_a)
    st.audio(file_b)
    path_a = convert_to_wav(file_a)
    path_b = convert_to_wav(file_b)
    ya, sr_a = librosa.load(path_a, sr=None)
    yb, sr_b = librosa.load(path_b, sr=None)
    fa = analyze_features(ya, sr_a)
    fb = analyze_features(yb, sr_b)

    st.subheader("📊 指標の比較")
    diffs = compare_features(fa, fb)
    for k, v in diffs.items():
        st.markdown(f"**{k}**: {v}")

    st.subheader("📝 コメントによる比較")
    st.info(compare_feedback(fa, fb))

    st.subheader("📈 時系列グラフ：音量 / ピッチ / 明瞭度 / 振幅")
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=False)

    t_rms_a = librosa.times_like(fa["rms"], sr=sr_a)
    t_rms_b = librosa.times_like(fb["rms"], sr=sr_b)
    t_pitch_a = librosa.times_like(fa["pitch"], sr=sr_a)
    t_pitch_b = librosa.times_like(fb["pitch"], sr=sr_b)
    t_flat_a = librosa.times_like(fa["clarity"], sr=sr_a)
    t_flat_b = librosa.times_like(fb["clarity"], sr=sr_b)
    t_amp_a = np.linspace(0, len(ya) / sr_a, len(ya))
    t_amp_b = np.linspace(0, len(yb) / sr_b, len(yb))

    axes[0].plot(t_rms_a, fa["rms"], label="A", color="blue")
    axes[0].plot(t_rms_b, fb["rms"], label="B", color="orange", linestyle="--")
    axes[0].set_ylabel("RMS (Volume)")
    axes[0].legend()

    axes[1].plot(t_pitch_a, fa["pitch"], label="A", color="blue")
    axes[1].plot(t_pitch_b, fb["pitch"], label="B", color="orange", linestyle="--")
    axes[1].set_ylabel("Pitch (Hz)")
    axes[1].legend()

    axes[2].plot(t_flat_a, fa["clarity"], label="A", color="blue")
    axes[2].plot(t_flat_b, fb["clarity"], label="B", color="orange", linestyle="--")
    axes[2].set_ylabel("Spectral Flatness")
    axes[2].legend()

    axes[3].plot(t_amp_a, ya, label="A", alpha=0.6, color="blue")
    axes[3].plot(t_amp_b, yb, label="B", alpha=0.6, color="orange")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlabel("Time (sec)")
    axes[3].legend()

    st.pyplot(fig)

    st.markdown("""
**🧾 グラフのラベル説明：**

- **RMS (Volume)**：音量（振幅の平均強度）  
- **Pitch (Hz)**：基本周波数（声の高さ）  
- **Spectral Flatness**：明瞭度の指標（ノイズ的かどうか）  
- **Amplitude**：生波形の振幅（瞬間的な変動）
""")

# ===== 区間分析セクション（音声ありの場合のみ表示） =====

def generate_natural_feedback(f1, f2, centroid_mean, bandwidth_mean, slope, flatness_mean):
    feedback = []
    if f1 and f2:
        if f1 > 800 or f2 < 1000:
            feedback.append("母音の明瞭度が低く、発音がこもっている可能性があります。")
    if centroid_mean < 1200:
        feedback.append("音声全体がこもった印象を与えるスペクトル分布です。")
    elif centroid_mean > 2500:
        feedback.append("明瞭で鋭い印象の音声特性が見られます。")
    if bandwidth_mean < 300:
        feedback.append("周波数の広がりが狭く、やや単調な音質です。")
    if slope < -10:
        feedback.append("高域の減衰が強く、声の抜けが弱く感じられる可能性があります。")
    if flatness_mean > 0.85:
        feedback.append("ハーモニック成分が少なく、ノイズ的な傾向が強いです。")
    return "📝 音響フィードバック:\n" + "\n".join(f"- {line}" for line in feedback) if feedback else "音響指標に大きな異常は見られません。"

# ✅ 音声ファイルが存在するか確認
if wav_audio is not None and len(wav_audio) > 0:
    st.header("↔️ 範囲指定して分析")
    st.markdown("#### 🧭 区間選択スライダー（開始位置を指定）")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav.write(wav_audio)
        tmp_wav.flush()
        y_full, sr_full = librosa.load(tmp_wav.name, sr=None)
        dur_full = librosa.get_duration(y=y_full, sr=sr_full)

    if dur_full < 15.0:
        st.warning("⚠️ 録音が15秒未満のため、区間分析は実行できません。")
    else:
        start_sec = st.slider("分析する開始位置（秒）", 0.0, dur_full - 15.0, step=0.1, format="%.1f 秒")
        start_sample = int(start_sec * sr_full)
        end_sample = int(start_sample + 15 * sr_full)
        y_seg = y_full[start_sample:end_sample]

        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(tmp_path, y_seg, sr_full)
        with open(tmp_path, "rb") as f:
            seg_bytes = f.read()
        st.audio(seg_bytes, format="audio/wav")

        if st.button("🔍 この15秒区間を分析する"):
            feat = analyze_features(y_seg, sr_full)

            st.subheader("📊 区間の指標（15秒間）")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔊 平均音量", f"{feat['rms_mean']:.4f}")
            col2.metric("🎵 平均ピッチ", f"{feat['pitch_mean']:.2f}Hz")
            col3.metric("🗣 明瞭度", f"{feat['clarity_mean']:.4f}")
            st.info(generate_feedback(feat))

            st.subheader("📈 区間の音量・ピッチ推移")
            fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            t_rms = librosa.times_like(feat["rms"], sr=sr_full)
            t_pitch = librosa.times_like(feat["pitch"], sr=sr_full)
            ax[0].plot(t_rms[: len(feat["rms"])], feat["rms"], color="blue")
            ax[0].set_ylabel("RMS")
            ax[1].plot(t_pitch[: len(feat["pitch"])], feat["pitch"], color="green")
            ax[1].set_ylabel("Pitch (Hz)")
            ax[1].set_xlabel("Time(s)")
            st.pyplot(fig)

            st.subheader("🧪 拡張音響指標")

            mfcc = librosa.feature.mfcc(y=y_seg, sr=sr_full, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            mid = len(y_seg) // 2
            frame = y_seg[mid - 512: mid + 512] * np.hamming(1024)
            lpc_order = int(sr_full / 1000) + 2
            lpc_coeffs = librosa.lpc(frame, order=lpc_order)
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) >= 0]
            freqs = np.angle(roots) * (sr_full / (2 * np.pi))
            formants = np.sort(freqs[freqs < sr_full / 2])
            f1 = round(formants[0], 1) if len(formants) > 0 else None
            f2 = round(formants[1], 1) if len(formants) > 1 else None

            centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr_full)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=y_seg, sr=sr_full)[0]
            flatness = librosa.feature.spectral_flatness(y=y_seg)[0]
            centroid_mean = round(np.mean(centroid), 1)
            bandwidth_mean = round(np.mean(bandwidth), 1)
            flatness_mean = round(np.mean(flatness), 4)

            S = np.abs(librosa.stft(y_seg, n_fft=1024))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            freqs_db = librosa.fft_frequencies(sr=sr_full)
            freqs_db = freqs_db[: S_db.shape[0]]
            mean_spectrum = np.mean(S_db, axis=1)
            slope = np.polyfit(freqs_db, mean_spectrum, deg=1)[0]

            st.markdown(f"- **フォルマント周波数**：F1 = {f1} Hz, F2 = {f2} Hz")
            st.markdown(f"- **スペクトル中心周波数**：{centroid_mean:.2f} Hz")
            st.markdown(f"- **スペクトル帯域幅**：{bandwidth_mean:.2f} Hz")
            st.markdown(f"- **スペクトル傾斜**：{slope:.2f} dB/oct")
            st.markdown(f"- **スペクトル平坦度**：{flatness_mean:.3f}")

            st.markdown("#### 🗒 音響のフィードバック")
            st.info(generate_natural_feedback(f1, f2, centroid_mean, bandwidth_mean, slope, flatness_mean))

            st.markdown("##### MFCC Mean & Variation Radar Chart")

            def plot_combined_radar(mean_vals, std_vals):
                labels = [f"MFCC{i+1}" for i in range(len(mean_vals))]
                angles = np.linspace(0, 2 * np.pi, len(mean_vals), endpoint=False).tolist()
                mean_vals += mean_vals[:1]
                std_vals += std_vals[:1]
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax.plot(angles, mean_vals, color="blue", linewidth=2, label="Mean")
                ax.fill(angles, mean_vals, color="blue", alpha=0.25)
                ax.plot(angles, std_vals, color="orange", linewidth=2, label="Variation")
                ax.fill(angles, std_vals, color="orange", alpha=0.25)
                ax.set_thetagrids(np.degrees(angles[:-1]), labels)
                ax.set_title("MFCC Mean & Variation (Normalized)")
                ax.grid(True)
                ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))
                st.pyplot(fig)

            mfcc_mean_norm = (mfcc_mean - np.min(mfcc_mean)) / (np.max(mfcc_mean) - np.min(mfcc_mean) + 1e-6)
            mfcc_std_norm = (mfcc_std - np.min(mfcc_std)) / (np.max(mfcc_std) - np.min(mfcc_std) + 1e-6)
            plot_combined_radar(mfcc_mean_norm.tolist(), mfcc_std_norm.tolist())

            st.markdown("""
**🧾 MFCCラベルの説明：**

- **MFCC1〜13** は音声スペクトルの形状を要約した特徴量です  
- **MFCC Mean** は平均的な音響特性を示し、声質や母音分布の傾向  
- **MFCC Variation** は音響の変動性（声の揺らぎや多様性）を示します  

※ 青＝MFCC平均  オレンジ＝MFCC変動（同一グラフ内に重ねて表示）
""")

with st.expander("🧾 数値の解釈を表示"):
    st.markdown("""
以下は、音声分析で表示される各数値項目の目安・意味・活用ポイントをまとめたものです。  
※医学的診断を目的としたものではなく、話し方の傾向を知るための参考情報です。

---

### 🔊 平均音量（RMS）
- **目安と範囲**：<span style='color:red;'>-25～-15 dBFS</span> が自然な会話音量  
- **解釈**：-30 dBFS 以下は非常に小さく聞こえ、-10 dBFS 以上は歪みの可能性も  
- **参考**：[OTO-to-MIMI：RMSと音圧](https://oto-to-mimi.com/mixing/about-peak-and-rms/)  
- **備考**：RMS は一時的なピークではなく、平均的な音量感（聴感上のボリューム）を表します。

---

### 🎵 平均ピッチ（Fundamental Frequency）  
- **目安と範囲**：<span style='color:red;'>男性：85～180 Hz、女性：165～255 Hz</span>  
- **解釈**：高い声は活発で若々しく、低い声は落ち着いた印象を与えやすい  
- **参考**：[宇都木研究室：音声のF0分析](https://sites.google.com/site/utsakr/Home/praat/f0)  
- **備考**：ピッチは緊張や感情で一時的に上下する場合があります。

---

### 🗣 明瞭度（Clarity）  
- **目安と範囲**：<span style='color:red;'>0.5～0.9（1に近いほど明瞭）</span>  
- **解釈**：0.6 未満はこもった音、0.8 以上はハッキリした発音傾向  
- **備考**：雑音・反響が多い環境では数値が下がりやすくなります。

---

### ⏱️ 話速（Speech Rate）  
- **目安と範囲**：<span style='color:red;'>7～9モーラ/秒（約0.14〜0.18秒/モーラ）</span>  
- **解釈**：8以上だと速く感じられ、6未満はゆっくり話す印象  
- **参考**：[J-STAGE：会話スタイルによる話速差](https://www.jstage.jst.go.jp/article/asjsc/4/1/4_SC-2024-13/_article/-char/ja)  
- **備考**：緊張時・読み上げ時・対話状況によって変動します。

---

### 📈 F0平均（基本周波数）  
- **目安と範囲**：<span style='color:red;'>男性：100〜130 Hz 女性：190〜230 Hz</span>（※話者平均値）  
- **解釈**：F0 が高い場合は、活気・不安・興奮と関連することも  
- **備考**：個人差が大きいため、相対比較が有効です。

---

### 🎙 フォルマント周波数（F1, F2, F3）
- **目安と範囲**：<span style='color:red;'>F1：300～900 Hz F2：900～2500 Hz</span>（母音によって変化）  
- **解釈**：F1＝口の開き F2＝舌の前後位置に関係し、母音の特徴を形成  
- **参考**：[東京外大：母音フォルマント分析](https://tufs.repo.nii.ac.jp/record/6449/files/acs082006_ful.pdf)  
- **備考**：発音訓練や自声モニタリングに役立つ項目

---

### 🔍 スペクトル中心周波数（Spectral Centroid）  
- **目安と範囲**：<span style='color:red;'>1500〜2500 Hz</span>  
- **解釈**：高い値＝シャープ／明るい印象 低い値＝こもった／暗い印象  
- **備考**：子音成分が多い声や、高域にエネルギーが偏っていると高くなります

---

### 🔍 スペクトル帯域幅（Spectral Bandwidth）  
- **目安と範囲**：<span style='color:red;'>300〜1500 Hz</span>  
- **解釈**：狭い→単調で落ち着いた 広い→豊かで開放的な音質  
- **備考**：母音の種類、声量の変化にも影響されます

---

### 🔍 スペクトル傾斜（Spectral Slope）  
- **目安と範囲**：<span style='color:red;'>-5 〜 -15 dB/oct</span>  
- **解釈**：-10以下は高域が大きく減衰 → 暗く落ち着いた音質  
- **参考**：[J-STAGE：声門源とスペクトル傾斜](https://www.jstage.jst.go.jp/article/asjsc/1/1/1_SC-2021-7/_article/-char/ja)  
- **備考**：加齢や声帯の状態変化で傾向が変わることがあります

---

### 🔍 スペクトル平坦度（Spectral Flatness）  
- **目安と範囲**：<span style='color:red;'>0.2〜0.6（0〜1の正規化値）</span>  
- **解釈**：低い値→ハーモニック（音色的） 高い値→ノイジー（雑音的）  
- **参考**：[J-GLOBAL：スペクトル平坦性研究](https://jglobal.jst.go.jp/detail?JGLOBAL_ID=202002255296422046)  
- **備考**：ボイスチェンジ検出や雑音評価にも用いられます
""", unsafe_allow_html=True)
