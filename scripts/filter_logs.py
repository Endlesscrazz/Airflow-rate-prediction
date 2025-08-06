# compile_logs.py
"""
A simple and robust script to scan a directory of log files, extract the most
important lines, and compile them into a single, concise output file.

This script is designed to be run from the project's root directory.
"""

import os
import argparse
import sys

# --- List of simple keywords to identify important lines ---
# This approach is more robust than complex regex for this log format.
IMPORTANT_KEYWORDS = [
    # Job/Fold Markers
    "--- Starting Training for Fold",
    "----- RUNNING FOLD",
    "----- Finished Fold",
    # Final Performance Metrics
    "Final R²",
    # Errors and Tracebacks
    "Traceback (most recent call last)",
    "Error",
    # You can add other important keywords here if needed
    # e.g., "Best Model Type from Nested CV"
]

def compile_log_files(log_dir, output_file):
    """
    Scans all .txt, .out, and .log files in a directory, filters them for
    important lines, and writes the result to a single output file.
    """
    found_log_files = []
    # First, find all relevant log files in the directory and its subdirectories
    for root, _, files in os.walk(log_dir):
        for fname in sorted(files):
            if fname.endswith(('.txt', '.out', '.log')):
                found_log_files.append(os.path.join(root, fname))

    if not found_log_files:
        print(f"Warning: No log files (.txt, .out, .log) found in '{log_dir}'.")
        return

    print(f"Found {len(found_log_files)} log files. Compiling important lines into '{output_file}'...")

    # Now, process each found log file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for log_path in found_log_files:
            try:
                # Write a clear header for each new file's content
                out_f.write(f"\n{'='*25} START OF LOG: {log_path} {'='*25}\n\n")
                
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                    for line in in_f:
                        # Check if any of the keywords are in the current line
                        if any(keyword in line for keyword in IMPORTANT_KEYWORDS):
                            out_f.write(line)
                            
                out_f.write(f"\n{'='*26} END OF LOG: {log_path} {'='*26}\n")

            except Exception as e:
                error_message = f"*** ERROR: Could not process file {log_path}: {e} ***\n"
                print(error_message, file=sys.stderr)
                out_f.write(error_message)

    print("Compilation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile important lines from multiple log files into a single, token-efficient file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "log_dir",
        help="Directory to scan for log files (e.g., 'logs/my_experiment/')."
    )
    parser.add_argument(
        "-o", "--output_filename",
        default="filtered_logs.txt",
        help="The name of the output file to be created inside the log directory."
    )
    args = parser.parse_args()

    # --- THIS IS THE FIX ---
    # Convert the user-provided relative path into an absolute path.
    # This makes the script work correctly regardless of how it's executed.
    absolute_log_dir = os.path.abspath(args.log_dir)
    # --- END OF FIX ---

    if not os.path.isdir(absolute_log_dir):
        # Use the corrected, absolute path in the error message for clarity
        print(f"Error: Log directory not found at '{absolute_log_dir}'", file=sys.stderr)
        sys.exit(1)
        
    # Use the corrected, absolute path for all subsequent operations
    output_path = os.path.join(absolute_log_dir, args.output_filename)
    compile_log_files(absolute_log_dir, output_path)

   # python src_cnn/filter_logs.py CNN_Results/iter-5/logs