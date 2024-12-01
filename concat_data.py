import os
import sys
import time
import pandas as pd
import gc
from audio_midi_pipeline import process_files

# Ensure the 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Base directory where your dataset is stored
base_dir = 'maestro-v3.0.0'  # Replace with your actual base directory

# List of years (directories) to process in order
years = ['2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017', '2018']

def shutdown_vm():
    # Function to shut down the virtual machine
    print("Shutting down the virtual machine in 60 seconds...")
    time.sleep(60)
    os.system("sudo shutdown -h now")
    sys.exit(1)

def save_dataframes(file_list, batch_size=130):
    df_list = []
    batch_number = 1
    total_files = len(file_list)
    iterations = 0

    for path in file_list:
        iterations += 1
        try:
            # Call your process_files function
            df = process_files(path)
            print(f"Processed File {iterations}/{total_files}: {path}")
            df_list.append(df)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Stop the virtual machine if an error occurs
            print("An error occurred during processing. Shutting down the virtual machine.")
            shutdown_vm()

        # Save batch when batch_size is reached or it's the last file
        if len(df_list) >= batch_size or iterations == total_files:
            # Concatenate and save the batch
            batch_df = pd.concat(df_list, ignore_index=True)
            batch_filename = f'master_dataframe_batch_{batch_number}.parquet'
            try:
                # Save Parquet file using Zappy compression
                batch_df.to_parquet(f"data/{batch_filename}", engine='pyarrow', compression='zstd')
                print(f"Saved Batch {batch_number} to {batch_filename}")
            except Exception as e:
                print(f"Error saving batch {batch_number}: {e}")
                # Stop the virtual machine if an error occurs
                print("An error occurred during saving. Shutting down the virtual machine.")
                shutdown_vm()
            finally:
                # Clear memory
                del df_list[:]
                del batch_df
                gc.collect()

            batch_number += 1

    return

try:
    # Collect all file paths across all years
    file_paths = []

    for year in years:
        year_dir = os.path.join(base_dir, year)

        # Check if the directory exists
        if os.path.isdir(year_dir):
            # Collect all relevant file paths in this year's directory
            for filename in os.listdir(year_dir):
                # Process only files that end with .midi or .wav
                if filename.endswith('.midi') or filename.endswith('.wav'):
                    # Remove the extension but keep the trailing period
                    file_base = os.path.splitext(filename)[0] + '.'
                    # Create the full path excluding the extension
                    file_path = os.path.join(year_dir, file_base)
                    file_paths.append(file_path)
        else:
            print(f"Directory {year_dir} does not exist.")
            continue  # Skip to the next year

    # Remove duplicates in file paths
    file_paths = list(set(file_paths))

    # Sort the file paths if needed
    file_paths.sort()

    # Process all files
    print(f"\nProcessing a total of {len(file_paths)} files across all years.\n")
    save_dataframes(file_paths, batch_size=130)

    # Force garbage collection to free up memory
    gc.collect()

except KeyboardInterrupt:
    print("Process was killed by user. Shutting down the virtual machine.")
    shutdown_vm()

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("Shutting down the virtual machine.")
    shutdown_vm()

shutdown_vm()
