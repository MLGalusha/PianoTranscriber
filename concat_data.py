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
    df_train_list = []
    df_test_list = []
    batch_file_list = []
    batch_number = 1
    total_files = len(file_list)
    iterations = 0

    for path in file_list:
        iterations += 1
        batch_file_list.append(path)

        # When batch_size is reached or it's the last file, process the batch
        if len(batch_file_list) >= batch_size or iterations == total_files:
            # Select 3 files for testing per batch
            test_file_paths = batch_file_list[:3]  # Take the first 3 files for testing
            train_file_paths = batch_file_list[3:]  # Remaining files for training

            processed_files = iterations - len(batch_file_list)
            for idx, file_path in enumerate(batch_file_list):
                try:
                    df = process_files(file_path)
                    processed_files += 1
                    print(f"Processed File {processed_files}/{total_files}: {file_path}")
                    if file_path in test_file_paths:
                        df_test_list.append(df)
                    else:
                        df_train_list.append(df)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    # Stop the virtual machine if an error occurs
                    print("An error occurred during processing. Shutting down the virtual machine.")
                    shutdown_vm()

            # Save training data batch
            if df_train_list:
                batch_df = pd.concat(df_train_list, ignore_index=True)

                # Add padding to not lose anything during training
                padding = pd.DataFrame(0, index=range(80), columns=batch_df.columns)
                batch_df = pd.concat([padding, batch_df], ignore_index=True)
                batch_df = pd.concat([batch_df, padding], ignore_index=True)

                batch_filename = f'train_batch_{batch_number}.parquet'
                try:
                    # Save Parquet file using ZSTD compression
                    batch_df.to_parquet(f"data/{batch_filename}", engine='pyarrow', compression='zstd')
                    print(f"Saved Training Batch {batch_number} to {batch_filename}")
                except Exception as e:
                    print(f"Error saving batch {batch_number}: {e}")
                    # Stop the virtual machine if an error occurs
                    print("An error occurred during saving. Shutting down the virtual machine.")
                    shutdown_vm()
                finally:
                    # Clear memory
                    del df_train_list[:]
                    del batch_df
                    gc.collect()

                batch_number += 1

            # Clear batch_file_list for the next batch
            batch_file_list = []

    # After all batches are processed, save the testing data
    if df_test_list:
        test_df = pd.concat(df_test_list, ignore_index=True)

        # Add padding to testing data as well
        padding = pd.DataFrame(0, index=range(80), columns=test_df.columns)
        test_df = pd.concat([padding, test_df], ignore_index=True)
        test_df = pd.concat([test_df, padding], ignore_index=True)

        test_filename = 'test_data.parquet'
        try:
            test_df.to_parquet(f"data/{test_filename}", engine='pyarrow', compression='zstd')
            print(f"Saved Testing Data to {test_filename}")
        except Exception as e:
            print(f"Error saving testing data: {e}")
            # Stop the virtual machine if an error occurs
            print("An error occurred during saving testing data. Shutting down the virtual machine.")
            shutdown_vm()
        finally:
            # Clear memory
            del df_test_list[:]
            del test_df
            gc.collect()

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
