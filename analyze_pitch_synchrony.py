import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

# Load the frame-level dataset
df = pd.read_csv("./data/frame_level_voice_features.csv")

# Set seaborn style
sns.set(style="whitegrid")

# Parameters
couple_id = "1"       # Change as needed
state = "baseline"    # "baseline" or "aroused"
pitch_col = "pitch__F0finEnv_sma"  # You can pick others from df.columns

# Filter the data
df_filtered = df[(df['couple_id'] == couple_id) & (df['state'] == state)]

# Separate male and female pitch time series
male_pitch = df_filtered[df_filtered['sex'] == 'male'][pitch_col].dropna().reset_index(drop=True)
female_pitch = df_filtered[df_filtered['sex'] == 'female'][pitch_col].dropna().reset_index(drop=True)

# Make sure they're same length for fair comparison
min_len = min(len(male_pitch), len(female_pitch))
male_pitch = male_pitch[:min_len]
female_pitch = female_pitch[:min_len]

# 1. Plot pitch curves
plt.figure(figsize=(12, 5))
plt.plot(male_pitch, label='Male Pitch', color='blue')
plt.plot(female_pitch, label='Female Pitch', color='red', linestyle='--')
plt.title(f"Pitch Contours - Couple {couple_id} - {state.capitalize()}")
plt.xlabel("Frame")
plt.ylabel("F0 (Hz)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Cross-correlation
corr = correlate(male_pitch - male_pitch.mean(), female_pitch - female_pitch.mean(), mode='full')
lags = np.arange(-len(male_pitch)+1, len(male_pitch))
max_corr = np.max(corr)
best_lag = lags[np.argmax(corr)]

plt.figure(figsize=(10, 4))
plt.plot(lags, corr)
plt.axvline(best_lag, color='red', linestyle='--', label=f'Max Corr Lag = {best_lag}')
plt.title(f"Cross-Correlation (lag) - Couple {couple_id} - {state.capitalize()}")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.legend()
plt.tight_layout()
plt.show()

print(f"🔁 Max cross-correlation: {max_corr:.3f} at lag {best_lag}")

# 3. DTW (Dynamic Time Warping)
dtw_distance, _ = fastdtw(male_pitch, female_pitch, dist=euclidean)
print(f"🧭 DTW distance (pitch): {dtw_distance:.2f}")

# 4. Pearson correlation
r, p = pearsonr(male_pitch, female_pitch)
print(f"🎲 Pearson correlation: r = {r:.3f}, p = {p:.3g}")
