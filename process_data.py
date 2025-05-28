import pandas as pd  # Import the pandas library for data handling



# Define the keywords for the features we are interested in
keywords = ['F0finEnv', 'jitter', 'shimmer', 'loudness']


# Define a function to process a CSV file and generate two output files:
# - One with raw summary statistics in CSV format
# - One with a human-readable (YAML-style) summary in plain text
def process_file(input_file, output_fileRaw, output_fileRead):

    # Read the input CSV file using semicolon as the separator
    df = pd.read_csv(input_file, sep=';')

    # Remove the 'frameTime' column if it exists, as we are not interested in time-based rows
    if 'frameTime' in df.columns:
        df = df.drop(columns=['frameTime'])

    # Remove the 'frameTime' column if it exists, as we are not interested in time-based rows
    selected_columns = df.columns[df.columns.str.contains('|'.join(keywords), case=False)]
    df = df[selected_columns]

    # Compute summary statistics: mean, std deviation, min, max for each selected feature
    summary_stats = df.agg(['mean', 'std', 'min', 'max']).transpose()

    
    # Save the summary statistics to a raw CSV file
    summary_stats.to_csv(output_fileRaw)



    # Build a nested dictionary: each feature has a sub-dictionary with its stats
    nested_stats = {
        column: {
            'mean': float(summary_stats.loc[column, 'mean']),
            'std': float(summary_stats.loc[column, 'std']),
            'min': float(summary_stats.loc[column, 'min']),
            'max': float(summary_stats.loc[column, 'max']),
        }
        for column in summary_stats.index
    }


    # Save the nested dictionary in a human-readable YAML-like format
    with open(output_fileRead, 'w') as f:
        for key, stats in nested_stats.items():
            f.write(f"{key}:\n")
            for stat_name, value in stats.items():
                f.write(f"  {stat_name}: {value}\n")
            f.write("\n") 

    # Print confirmation with paths to output files
    print("\n Speaker summary saved in raw format to"+output_fileRaw+" and in readable format in "+output_fileRead+"\n")


process_file("./input/featuresBaseline_speakerMan.csv", "./results/raw/resultsMan.csv", "./results/read/resultsMan.txt")
process_file("./input/featuresBaseline_speakerWoman.csv", "./results/raw/resultsWoman.csv", "./results/read/resultsWoman.txt")