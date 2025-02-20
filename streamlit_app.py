import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
from io import BytesIO
from scipy.signal import butter, lfilter, sawtooth
import json

SAMPLE_RATE = 44100

# --- DEFAULT PARAMETERS ---
DEFAULTS = {
    "duration_ms": 20,          # Click duration in milliseconds
    "waveform": "Sine",         # Base waveform: Sine, Square, Sawtooth, Triangle
    "frequency": 2000,          # Base frequency in Hz
    "amplitude": 0.5,           # 0..1
    "overtone_enabled": True,   # Enable overtone
    "overtone_freq": 3000,      # Overtone frequency in Hz
    "overtone_amp": 0.2,        # Overtone amplitude (0..1)

    # ADSR envelope (in ms, sustain is a level 0..1)
    "attack_ms": 1,
    "decay_ms": 5,
    "sustain_level": 0.5,
    "release_ms": 5,

    # Mechanical resonance (exponential decay factor)
    "mechanical_decay": 0.02,
    # Noise amplitude
    "noise_amp": 0.002,

    # Bitcrusher effect
    "bitcrusher_enabled": False,
    "bit_depth": 8,

    # Reverb effect
    "reverb_enabled": False,
    "reverb_decay": 0.2,   # seconds
    "reverb_mix": 0.3,     # 0..1

    # Filter effect
    "filter_enabled": False,
    "filter_type": "lowpass",     # Options: "lowpass", "highpass", "bandpass"
    "filter_cutoff_low": 500,       # For bandpass: lower cutoff (Hz)
    "filter_cutoff_high": 2000,     # For bandpass: upper cutoff (Hz)
    "filter_cutoff_single": 2000    # For lowpass/highpass (Hz)
}

# --- AUDIO PROCESSING FUNCTIONS ---

def generate_waveform(waveform, freq, amp, t):
    w = waveform.lower()
    if w == "sine":
        return amp * np.sin(2 * np.pi * freq * t)
    elif w == "square":
        return amp * np.sign(np.sin(2 * np.pi * freq * t))
    elif w == "sawtooth":
        return amp * sawtooth(2 * np.pi * freq * t)
    elif w == "triangle":
        return amp * (2 * np.abs(sawtooth(2 * np.pi * freq * t, width=0.5)) - 1)
    else:
        return amp * np.sin(2 * np.pi * freq * t)  # fallback to sine

def apply_adsr(signal, attack_ms, decay_ms, sustain_level, release_ms, fs):
    """
    Applies an ADSR envelope to the signal.
    All times are in milliseconds.
    """
    length = len(signal)
    env = np.ones(length, dtype=np.float32)
    a = int(fs * (attack_ms / 1000.0))
    d = int(fs * (decay_ms / 1000.0))
    r = int(fs * (release_ms / 1000.0))
    total = a + d + r
    s = max(0, length - total)

    end_a = min(a, length)
    if end_a > 0:
        env[:end_a] = np.linspace(0, 1, end_a, endpoint=False)
    start_d = end_a
    end_d = start_d + min(d, length - start_d)
    if end_d > start_d:
        env[start_d:end_d] = np.linspace(1, sustain_level, end_d - start_d, endpoint=False)
    start_s = end_d
    end_s = start_s + min(s, length - start_s)
    if end_s > start_s:
        env[start_s:end_s] = sustain_level
    start_r = end_s
    end_r = start_r + min(r, length - start_r)
    if end_r > start_r:
        env[start_r:end_r] = np.linspace(sustain_level, 0, end_r - start_r, endpoint=False)
    return signal * env

def apply_mechanical_resonance(signal, decay_rate):
    length = len(signal)
    exp_env = np.exp(-np.linspace(0, decay_rate, length))
    return signal * exp_env

def apply_bitcrusher(signal, bit_depth):
    factor = 2 ** bit_depth
    return np.round(signal * factor) / factor

def apply_reverb(signal, fs, decay_time, mix):
    impulse_len = int(fs * decay_time)
    if impulse_len < 1:
        return signal
    impulse = np.exp(-np.linspace(0, 3, impulse_len))
    impulse /= np.sum(impulse)
    wet = np.convolve(signal, impulse, mode='full')[:len(signal)]
    return (1 - mix) * signal + mix * wet

def apply_filter(signal, fs, ftype, cutoff_low, cutoff_high, cutoff_single):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    if ftype == "bandpass":
        if cutoff_low >= cutoff_high:
            st.warning("For bandpass filter, low cutoff must be lower than high cutoff. Filter not applied. ⚠️")
            return signal
        low = cutoff_low / nyq
        high = cutoff_high / nyq
        low = max(low, 0.0001)
        high = min(high, 0.9999)
        b, a = butter(2, [low, high], btype='band', analog=False)
    else:
        norm_cut = cutoff_single / nyq
        norm_cut = min(max(norm_cut, 0.0001), 0.9999)
        b, a = butter(2, norm_cut, btype=ftype, analog=False)
    return lfilter(b, a, signal)

def normalize_signal(signal):
    max_val = np.max(np.abs(signal))
    return signal if max_val == 0 else signal / max_val

def synthesize_click_sound(params):
    duration_ms = params["duration_ms"]
    waveform = params["waveform"]
    frequency = params["frequency"]
    amplitude = params["amplitude"]
    overtone_enabled = params["overtone_enabled"]
    overtone_freq = params["overtone_freq"]
    overtone_amp = params["overtone_amp"]

    attack_ms = params["attack_ms"]
    decay_ms = params["decay_ms"]
    sustain_level = params["sustain_level"]
    release_ms = params["release_ms"]

    mechanical_decay = params["mechanical_decay"]
    noise_amp = params["noise_amp"]

    bitcrusher_enabled = params["bitcrusher_enabled"]
    bit_depth = params["bit_depth"]

    reverb_enabled = params["reverb_enabled"]
    reverb_decay = params["reverb_decay"]
    reverb_mix = params["reverb_mix"]

    filter_enabled = params["filter_enabled"]
    filter_type = params["filter_type"]
    cutoff_low = params["filter_cutoff_low"]
    cutoff_high = params["filter_cutoff_high"]
    cutoff_single = params["filter_cutoff_single"]

    # Time array
    duration_sec = duration_ms / 1000.0
    length = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, length, endpoint=False)
    if length == 0:
        return np.zeros(0), t

    # Base waveform
    signal = generate_waveform(waveform, frequency, amplitude, t)

    # Overtone addition
    if overtone_enabled:
        signal += generate_waveform("sine", overtone_freq, overtone_amp, t)

    # Apply ADSR envelope
    signal = apply_adsr(signal, attack_ms, decay_ms, sustain_level, release_ms, SAMPLE_RATE)

    # Apply mechanical resonance
    signal = apply_mechanical_resonance(signal, mechanical_decay)

    # Add noise
    signal += noise_amp * np.random.randn(len(signal))

    # Bitcrusher effect
    if bitcrusher_enabled:
        signal = apply_bitcrusher(signal, bit_depth)

    # Reverb effect
    if reverb_enabled:
        signal = apply_reverb(signal, SAMPLE_RATE, reverb_decay, reverb_mix)

    # Filter effect
    if filter_enabled:
        signal = apply_filter(signal, SAMPLE_RATE, filter_type, cutoff_low, cutoff_high, cutoff_single)

    # Normalize the output
    signal = normalize_signal(signal)
    return signal, t

def get_audio_bytes(signal):
    buffer = BytesIO()
    sf.write(buffer, signal.astype(np.float32), SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    return buffer

# --- STREAMLIT APP ---

st.set_page_config(layout="centered")
st.title("🎵 Click Sound Designer")

st.markdown("""
Create short, pleasant click sounds by adjusting various parameters.  
Customize the waveform, frequency, overtone, ADSR envelope, mechanical resonance, noise,  
bitcrusher, reverb, and filter settings to craft the perfect click for your app. ✨
""")

# --- PRESETS: EXPORT/IMPORT SETTINGS ---
st.sidebar.header("💾 Presets")

# Import settings
preset_file = st.sidebar.file_uploader("Import Settings (JSON)", type=["json"])
if preset_file is not None:
    try:
        imported_settings = json.load(preset_file)
        st.sidebar.success("Settings imported successfully! ✅")
    except Exception as e:
        st.sidebar.error(f"Error importing settings: {e}")
else:
    imported_settings = None

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Basic Parameters")
duration_ms = st.sidebar.slider("Duration (ms)", 1, 200, DEFAULTS["duration_ms"])
waveform = st.sidebar.selectbox("Waveform", ["Sine", "Square", "Sawtooth", "Triangle"], index=0)
frequency = st.sidebar.slider("Base Frequency (Hz)", 50, 5000, DEFAULTS["frequency"])
amplitude = st.sidebar.slider("Amplitude (0-1)", 0.0, 1.0, DEFAULTS["amplitude"], 0.05)

st.sidebar.header("2. Overtone")
overtone_enabled = st.sidebar.checkbox("Enable Overtone", DEFAULTS["overtone_enabled"])
if overtone_enabled:
    overtone_freq = st.sidebar.slider("Overtone Frequency (Hz)", 50, 8000, DEFAULTS["overtone_freq"])
    overtone_amp = st.sidebar.slider("Overtone Amplitude (0-1)", 0.0, 1.0, DEFAULTS["overtone_amp"], 0.05)
else:
    overtone_freq = DEFAULTS["overtone_freq"]
    overtone_amp = DEFAULTS["overtone_amp"]

st.sidebar.header("3. Envelope (ADSR)")
attack_ms = st.sidebar.slider("Attack (ms)", 0, 100, DEFAULTS["attack_ms"])
decay_ms = st.sidebar.slider("Decay (ms)", 0, 200, DEFAULTS["decay_ms"])
sustain_level = st.sidebar.slider("Sustain (0-1)", 0.0, 1.0, DEFAULTS["sustain_level"], 0.05)
release_ms = st.sidebar.slider("Release (ms)", 0, 200, DEFAULTS["release_ms"])

st.sidebar.header("4. Mechanical & Noise")
mechanical_decay = st.sidebar.slider("Mechanical Decay", 0.0, 0.1, DEFAULTS["mechanical_decay"], 0.01)
noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 0.01, DEFAULTS["noise_amp"], 0.001)

st.sidebar.header("5. Bitcrusher & Reverb")
bitcrusher_enabled = st.sidebar.checkbox("Bitcrusher", DEFAULTS["bitcrusher_enabled"])
if bitcrusher_enabled:
    bit_depth = st.sidebar.slider("Bit Depth", 4, 16, DEFAULTS["bit_depth"])
else:
    bit_depth = DEFAULTS["bit_depth"]

reverb_enabled = st.sidebar.checkbox("Reverb", DEFAULTS["reverb_enabled"])
if reverb_enabled:
    reverb_decay = st.sidebar.slider("Reverb Decay (sec)", 0.1, 5.0, DEFAULTS["reverb_decay"], 0.1)
    reverb_mix = st.sidebar.slider("Reverb Mix (0-1)", 0.0, 1.0, DEFAULTS["reverb_mix"], 0.05)
else:
    reverb_decay = DEFAULTS["reverb_decay"]
    reverb_mix = DEFAULTS["reverb_mix"]

st.sidebar.header("6. Filter")
filter_enabled = st.sidebar.checkbox("Enable Filter", DEFAULTS["filter_enabled"])
if filter_enabled:
    filter_type = st.sidebar.selectbox("Filter Type", ["lowpass", "highpass", "bandpass"], index=0)
    if filter_type == "bandpass":
        filter_cutoff_low = st.sidebar.slider("Filter Low (Hz)", 50, 5000, DEFAULTS["filter_cutoff_low"], 50)
        filter_cutoff_high = st.sidebar.slider("Filter High (Hz)", 100, 10000, DEFAULTS["filter_cutoff_high"], 50)
        filter_cutoff_single = DEFAULTS["filter_cutoff_single"]
    else:
        filter_cutoff_single = st.sidebar.slider("Filter Cutoff (Hz)", 50, 10000, DEFAULTS["filter_cutoff_single"], 50)
        filter_cutoff_low = DEFAULTS["filter_cutoff_low"]
        filter_cutoff_high = DEFAULTS["filter_cutoff_high"]
else:
    filter_type = DEFAULTS["filter_type"]
    filter_cutoff_single = DEFAULTS["filter_cutoff_single"]
    filter_cutoff_low = DEFAULTS["filter_cutoff_low"]
    filter_cutoff_high = DEFAULTS["filter_cutoff_high"]

# Assemble current parameters from controls
current_params = {
    "duration_ms": duration_ms,
    "waveform": waveform,
    "frequency": frequency,
    "amplitude": amplitude,
    "overtone_enabled": overtone_enabled,
    "overtone_freq": overtone_freq,
    "overtone_amp": overtone_amp,

    "attack_ms": attack_ms,
    "decay_ms": decay_ms,
    "sustain_level": sustain_level,
    "release_ms": release_ms,

    "mechanical_decay": mechanical_decay,
    "noise_amp": noise_amp,

    "bitcrusher_enabled": bitcrusher_enabled,
    "bit_depth": bit_depth,

    "reverb_enabled": reverb_enabled,
    "reverb_decay": reverb_decay,
    "reverb_mix": reverb_mix,

    "filter_enabled": filter_enabled,
    "filter_type": filter_type,
    "filter_cutoff_low": filter_cutoff_low,
    "filter_cutoff_high": filter_cutoff_high,
    "filter_cutoff_single": filter_cutoff_single,
}

# If settings were imported, override current parameters
if imported_settings is not None:
    current_params = imported_settings
    st.sidebar.success("Imported settings applied!")

# Export current settings as JSON (download button)
st.sidebar.download_button("Export Settings (JSON) 📥",
                           data=json.dumps(current_params, indent=4),
                           file_name="click_settings.json",
                           mime="application/json")

# --- SOUND SYNTHESIS ---
signal, t = synthesize_click_sound(current_params)
audio_buffer = get_audio_bytes(signal)

# --- VISUALIZATION ---
st.subheader("⏱ Time-Domain Waveform")
fig_time = go.Figure()
fig_time.add_trace(go.Scatter(x=t, y=signal, mode='lines', name='Waveform'))
fig_time.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    template="plotly_white",
    width=600,
    height=300
)
st.plotly_chart(fig_time, use_container_width=False)

st.subheader("📈 Frequency Spectrum (FFT)")
if len(signal) > 0:
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(len(signal), 1.0/SAMPLE_RATE)
    mag = np.abs(fft_vals)
else:
    fft_freqs = [0]
    mag = [0]
fig_fft = go.Figure()
fig_fft.add_trace(go.Scatter(x=fft_freqs, y=mag, mode='lines', name='Spectrum'))
fig_fft.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="Magnitude",
    template="plotly_white",
    width=600,
    height=300
)
st.plotly_chart(fig_fft, use_container_width=False)

st.subheader("▶️ Playback & Download")
st.audio(audio_buffer, format="audio/wav")
st.download_button("Download WAV", data=audio_buffer, file_name="click_sound.wav", mime="audio/wav")
