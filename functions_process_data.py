import pandas as pd
import os
from scipy.stats import skew, kurtosis

# Folder where input CSV files are stored
FOLDER_PATH = "./input/"
# List all CSV files in the folder
csv_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]

# Define function to group features by type (e.g., pitch, jitter)
def features_by_group(df):
    return {
        "pitch": [col for col in df.columns if "F0finEnv" in col],
        "intensity": [col for col in df.columns if "pcm_loudness" in col],
        "jitter": [col for col in df.columns if "jitter" in col and "jitterDDP" not in col],
        "shimmer": [col for col in df.columns if "shimmer" in col],
        "speech_rate": [col for col in df.columns if "duration" in col or "risetime" in col or "falltime" in col],
        "timbre": [col for col in df.columns if "pcm_fftMag_mfcc" in col], 
        "energy": [col for col in df.columns if 'pcm_loudness' in col or 'variance' in col],  
        "temporal_alignment": [col for col in df.columns if 'ctime' in col or 'maxPos' in col or 'minPos' in col or 'upleveltime' in col or 'downleveltime' in col],
    }

# Extract metadata (couple ID, sex, and state) from the filename
def parse_filename(filename):
    base = filename.replace(".csv", "")
    parts = base.split("_")
    return {
        "couple_id": parts[0].replace("couple", ""),
        "sex": parts[1],
        "state": parts[2]
    }

# (Unused) Utility function to aggregate column-wise statistics
def aggregate_features(df, columns, prefix=""):
    aggs = {}
    for col in columns:
        if col in df.select_dtypes(include='number').columns:
            values = df[col].dropna()
            aggs[f"{prefix}{col}_mean"] = values.mean()
            aggs[f"{prefix}{col}_std"] = values.std()
            aggs[f"{prefix}{col}_min"] = values.min()
            aggs[f"{prefix}{col}_max"] = values.max()
    return aggs

# Create frame-level data by concatenating all individual frames from CSVs
def create_data():
    frame_level_data = []

    for file in csv_files:
        file_path = os.path.join(FOLDER_PATH, file)
        df = pd.read_csv(file_path, sep=";")
        meta = parse_filename(file)
        grouped = features_by_group(df)

        # Filter only relevant feature columns
        relevant_cols = [col for cols in grouped.values() for col in cols]
        df = df[relevant_cols]
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert all to numeric (non-numeric → NaN)

        # Rename columns to indicate feature group
        df_frame = df.copy()
        for group, columns in grouped.items():
            for col in columns:
                if col in df_frame.columns:
                    df_frame.rename(columns={col: f"{group}__{col}"}, inplace=True)

        # Add metadata to each frame
        df_frame['couple_id'] = meta['couple_id']
        df_frame['sex'] = meta['sex']
        df_frame['state'] = meta['state']
        frame_level_data.append(df_frame)

    # Combine all CSVs' frames into a single DataFrame
    frame_level_df = pd.concat(frame_level_data, ignore_index=True)

    # Save to CSV
    os.makedirs("./data", exist_ok=True)
    frame_level_df.to_csv("./data/frame_level_voice_features.csv", index=False)

    print("✅ Frame-level DataFrame created with shape:", frame_level_df.shape)
    return frame_level_df

# Create a summary-level dataset with mean, std, min, max for each feature per file
def create_data2():
    summary_data = []

    for file in csv_files:
        file_path = os.path.join(FOLDER_PATH, file)
        df = pd.read_csv(file_path, sep=";")
        meta = parse_filename(file)
        grouped = features_by_group(df)

        # Keep only relevant feature columns
        relevant_cols = [col for cols in grouped.values() for col in cols]
        df = df[relevant_cols]

        # Rename columns to include group prefix
        renamed_cols = {}
        for group, columns in grouped.items():
            for col in columns:
                if col in df.columns:
                    renamed_cols[col] = f"{group}__{col}"
        df.rename(columns=renamed_cols, inplace=True)

        # Compute aggregate statistics (mean, std, min, max) for each column
        agg_dict = {}
        for col in df.columns:
            agg_dict[f"{col}_mean"] = df[col].mean(skipna=True)
            agg_dict[f"{col}_std"] = df[col].std(skipna=True)
            agg_dict[f"{col}_min"] = df[col].min(skipna=True)
            agg_dict[f"{col}_max"] = df[col].max(skipna=True)

        # Add metadata
        agg_dict['couple_id'] = meta['couple_id']
        agg_dict['sex'] = meta['sex']
        agg_dict['state'] = meta['state']

        summary_data.append(agg_dict)

    # Combine all summaries into a DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save summary-level CSV
    os.makedirs("./data", exist_ok=True)
    summary_df.to_csv("./data/summary_voice_features.csv", index=False)

    print("✅ Summary DataFrame created with shape:", summary_df.shape)
    return summary_df
