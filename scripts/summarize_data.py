# scripts/summarize_data.py
"""
A utility script to load the combined ground truth CSV and print a 
comprehensive summary of the entire dataset.
"""
import os
import sys
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Point to the NEW combined file
COMBINED_GROUND_TRUTH_PATH = os.path.join(project_root, "airflow_ground_truth_combined.csv")

def main():
    """Main function to load and summarize the combined data."""
    if not os.path.exists(COMBINED_GROUND_TRUTH_PATH):
        print(f"Error: Combined ground truth file not found at '{COMBINED_GROUND_TRUTH_PATH}'")
        print("Please run `create_combined_ground_truth.py` first.")
        return

    print(f"--- Summarizing Data from '{COMBINED_GROUND_TRUTH_PATH}' ---")
    df = pd.read_csv(COMBINED_GROUND_TRUTH_PATH)

    print(f"\n--- Found {len(df)} Total Samples (Hole-Level) ---")
    
    print("\n--- Full Data Listing ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 150):
        print(df)
        
    print("\n--- Summary Statistics ---")
    
    print("\nCounts per Material:")
    print(df['material'].value_counts())

    print("\nCounts per Voltage:")
    print(df['voltage'].value_counts().sort_index())

    print("\nCounts per Delta_T:")
    print(df['delta_T'].value_counts().sort_index())
    
    print("\nNumber of unique videos:")
    print(df['video_id'].nunique())

if __name__ == "__main__":
    main()
# python -m scripts.summarize_data