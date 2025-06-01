import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the frame-level dataset
df = pd.read_csv("./data/frame_level_voice_features.csv")

# Set seaborn style
sns.set(style="whitegrid")

# Choose couple and state
couple_id = "1"        # Change as needed
state = "baseline"     # "aroused" or "baseline"

# Get pitch-related columns
pitch_cols = [col for col in df.columns if col.startswith("pitch__")]

# Filter the data
df_filtered = df[(df['couple_id'] == couple_id) & (df['state'] == state)]

# Separate male and female
df_male = df_filtered[df_filtered['sex'] == 'male'][pitch_cols].reset_index(drop=True)
df_female = df_filtered[df_filtered['sex'] == 'female'][pitch_cols].reset_index(drop=True)

# Plot the pitch contours
plt.figure(figsize=(14, 6))
for col in pitch_cols:
    plt.plot(df_male[col], label=f"Male - {col}", alpha=0.6)
    plt.plot(df_female[col], label=f"Female - {col}", linestyle='--', alpha=0.6)

plt.title(f"Pitch Contours - Couple {couple_id} - {state.capitalize()}")
plt.xlabel("Frame Index")
plt.ylabel("F0 Value")
plt.legend()
plt.tight_layout()
plt.show()
