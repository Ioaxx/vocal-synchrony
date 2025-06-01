from functions_process_data import *
from functions_graphs import *
from calculate_functions import *
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, f_oneway
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns



def main():
    data = create_data2()
    
    pitch_cols = [col for col in data.columns if col.startswith('pitch__')]
    intensity_cols = [col for col in data.columns if col.startswith('intensity__')]
    jitter_cols = [col for col in data.columns if col.startswith('jitter__')]
    shimmer_cols = [col for col in data.columns if col.startswith('shimmer__')]
    speech_rate_cols = [col for col in data.columns if col.startswith('speech_rate__')]
    timbre_cols = [col for col in data.columns if col.startswith('timbre__')]
    energy_cols = [col for col in data.columns if col.startswith('energy__')]
    temporal_alignment_cols = [col for col in data.columns if col.startswith('temporal_alignment__')]


    result= ttest_arousal_vs_baseline_for_pitch(data, timbre_cols)
    result_clean = simplify_cells(result)
    print(result_clean)

if __name__ == "__main__":
    main()