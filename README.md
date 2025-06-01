#Vocal Synchrony Data Tool

This project provides tools to analyze vocal synchrony, with a focus on pitch and other acoustic features across different emotional states (e.g., baseline vs. aroused). The code is structured for modular use and is intended to work within a Python virtual environment.
Project Structure and Workflow

    analyze_pitch_synchrony.py: Performs pitch synchrony analysis using a properly prepared dataset.

    calculate_functions.py: Initially intended for a broader range of statistical computations, but development paused due to data processing issues.

    graphs.py: Reserved for visualization of results once the dataset has been properly defined and processed.

    process_data.py: Contains functions for preprocessing and organizing the dataset to make it ready for analysis.

    Two main scripts (main.py, main_v2.py): Represent two different workflows/attempts for data processing and analysis.

Setup Instructions

    Create and activate a virtual environment:

python3 -m venv ./.venv
.\.venv\Scripts\activate.ps1   # On Windows PowerShell

Install the required dependencies:

pip install -r requirements.txt