# Here's the optimized version - replaces the problematic parts
from process_functions import *
from test import *
import pandas as pd
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load data with progress tracking
    print("Loading data...")
    load_and_save_processed_data()
    
    results = run_ttests_per_feature("./output/processed")

# View or save results
    results['female'].to_csv("female_ttest_results.csv", index=False)
    results['male'].to_csv("male_ttest_results.csv", index=False)

    

if __name__ == "__main__":
    main()
