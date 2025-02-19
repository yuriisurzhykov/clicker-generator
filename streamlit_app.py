import numpy as np
import streamlit as st
import soundfile as sf
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# --- CSS Styling ---
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .main {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007bff !important;
        color: white !important;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center; color: #007bff;">🔊 Ultimate Click Sound Designer</h1>',
            unsafe_allow_html=True)
st.markdown('<div class="main">', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("🎚 Sound Customization")

# Duration and frequency
duration = st.sidebar.slider("Duration (ms)", 1, 100, 10) / 1000
frequency = st.sidebar.slider("Base Frequency (Hz)", 500, 5000, 2000)
amplitude = st.sidebar.slider("Amplitude (0-1)", 0.1, 1.0, 0.5)

# Waveform Selection
waveform = st.sidebar.selectbox("Waveform Type", ["Sine", "Square", "Sawtooth", "Triangle"])

# Overtone Settings
st.sidebar.header("🎵 Overtone Settings")
overtone_freq = st.sidebar.slider("Overtone Frequency (Hz)", 500, 10000, 3000)
overtone_amplitude = st.sidebar.slider("Overtone Amplitude (0-1)", 0.0, 1.0, 0.2)

# Mechanical Click Effects
st.sidebar.header("🔩 Mechanical Properties")
mechanical_decay = st.sidebar.slider("Metallic Resonance (0-1)", 0.0, 1.0, 0.02)
noise_amplitude = st.sidebar.slider("Mechanical Noise Level (0-1)", min_value=0.000, max_value=0.01, step=0.005)

# Bitcrusher Effect
st.sidebar.header("🖥 Digital Click Effects")
bit_depth = st.sidebar.slider("Bit Depth (Lower = More Digital)", 4, 16, 8)

# ADSR Envelope
st.sidebar.header("🎛 Envelope (ADSR)")
attack_time = st.sidebar.slider("Attack Time (ms)", 0, 50, 1) / 1000
decay_time = st.sidebar.slider("Decay Time (ms)", 0, 100, 5) / 1000
sustain_level = st.sidebar.slider("Sustain Level (0-1)", 0.0, 1.0, 0.5)
release_time = st.sidebar.slider("Release Time (ms)", 0, 100, 10) / 1000

sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# --- Waveform Generation ---
if waveform == "Sine":
    click_sound = amplitude * np.sin(2 * np.pi * frequency * t)
elif waveform == "Square":
    click_sound = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
elif waveform == "Sawtooth":
    click_sound = amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
elif waveform == "Triangle":
    click_sound = amplitude * 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1

# Add overtone
click_sound += overtone_amplitude * np.sin(2 * np.pi * overtone_freq * t)


# --- ADSR Envelope ---
def adsr_envelope(signal):
    length = len(signal)
    env = np.ones(length)

    attack_samples = int(sample_rate * attack_time)
    decay_samples = int(sample_rate * decay_time)
    release_samples = int(sample_rate * release_time)

    env[:attack_samples] = np.linspace(0, 1, attack_samples)
    env[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
    env[-release_samples:] = np.linspace(sustain_level, 0, release_samples)

    return signal * env


click_sound = adsr_envelope(click_sound)


# --- Mechanical Resonance ---
def mechanical_resonance(signal, decay_rate):
    impulse_response = np.exp(-np.linspace(0, decay_rate, len(signal)))
    return signal * impulse_response


click_sound = mechanical_resonance(click_sound, mechanical_decay)

# --- Add Noise ---
click_sound += noise_amplitude * np.random.randn(len(click_sound))

# --- Bitcrusher Effect ---
click_sound = np.round(click_sound * (2 ** bit_depth)) / (2 ** bit_depth)

# --- Save and Display ---
sf.write("click_sound.wav", click_sound, sample_rate)

# --- Plot the Waveform ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=click_sound, mode='lines', name='Waveform'))
fig.update_layout(title="🔊 Click Sound Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude",
                  template="plotly_white")

st.plotly_chart(fig, use_container_width=True)

# --- Audio and Download ---
st.audio("click_sound.wav")
st.download_button("📥 Download WAV", "click_sound.wav", file_name="click_sound.wav")

st.markdown('</div>', unsafe_allow_html=True)
