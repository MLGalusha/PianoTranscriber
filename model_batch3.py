import os
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Define the CRNN model
class PitchDetectionCRNN(nn.Module):
    def __init__(self, num_pitches=88):
        super(PitchDetectionCRNN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Calculate the size after convolutional layers to define LSTM input size
        # Input size is (batch_size, channels=1, sequence_length=513)
        # After Conv and Pooling, sequence_length reduces
        self.sequence_length = 513 // (2 ** 3)  # Divided by 2 three times due to MaxPool1d
        self.lstm_input_size = 64  # Number of features after conv layers

        # Recurrent layers for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Fully connected layers for final classification
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 2, 256),  # 128 * 2 because of bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_pitches)
            # No activation here since we'll use BCEWithLogitsLoss
        )

    def forward(self, x):
        # x shape: (batch_size, 1, 513)

        # Convolutional layers
        x = self.conv_layers(x)
        # x shape after conv_layers: (batch_size, channels=64, sequence_length)

        # Prepare data for LSTM
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, sequence_length, features)
        # x shape: (batch_size, sequence_length, features=64)

        # Recurrent layers
        x, _ = self.lstm(x)
        # x shape: (batch_size, sequence_length, hidden_size*2)

        # Take the output from the last time step
        x = x[:, -1, :]  # Shape: (batch_size, hidden_size*2)

        # Fully connected layers
        x = self.fc_layers(x)
        return x  # Output raw logits

# Training function
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0  # Initialize batch counter

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Data validation checks for model outputs
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"NaN or Inf detected in model outputs in batch {batch_idx}")
            exit()

        # Compute loss
        loss = criterion(output, target)
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1  # Increment batch counter

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

    average_loss = total_loss / num_batches
    return average_loss

# Validation function
def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0  # Initialize batch counter

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Data validation checks for model outputs
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("NaN or Inf detected in model outputs during validation")
                exit()

            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1  # Increment batch counter

    average_loss = total_loss / num_batches
    return average_loss

# Evaluation functions
def evaluate_model(model, data_loader, device, threshold=0.5):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            # Apply Sigmoid activation to get probabilities
            outputs = torch.sigmoid(outputs)

            # Convert predictions to binary using threshold
            predictions = (outputs > threshold).float()

            # Move to CPU and convert to numpy for sklearn metrics
            predictions = predictions.cpu().numpy()
            target = target.cpu().numpy()

            all_predictions.append(predictions)
            all_targets.append(target)

    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Calculate metrics
    # Per-pitch metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, zero_division=0
    )

    # Overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_targets.flatten(), all_predictions.flatten(), average='binary', zero_division=0
    )

    # Calculate accuracy per pitch
    pitch_accuracy = (all_targets == all_predictions).mean(axis=0)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_targets.flatten(), all_predictions.flatten())

    # Create a DataFrame with all metrics
    metrics_df = pd.DataFrame({
        'Pitch': [f'Pitch_{i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': pitch_accuracy
    })

    # Add overall metrics
    overall_metrics = {
        'Overall Accuracy': overall_accuracy,
        'Overall Precision': overall_precision,
        'Overall Recall': overall_recall,
        'Overall F1': overall_f1
    }

    return metrics_df, overall_metrics

def print_evaluation_results(metrics_df, overall_metrics):
    print("\n=== Overall Model Performance ===")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n=== Top 5 Best Performing Pitches ===")
    print(metrics_df.nlargest(5, 'F1 Score')[['Pitch', 'Precision', 'Recall', 'F1 Score', 'Accuracy']])

    print("\n=== Top 5 Worst Performing Pitches ===")
    print(metrics_df.nsmallest(5, 'F1 Score')[['Pitch', 'Precision', 'Recall', 'F1 Score', 'Accuracy']])

    # Calculate performance distribution
    print("\n=== Performance Distribution ===")
    print("\nF1 Score Distribution:")
    print(metrics_df['F1 Score'].describe())

def evaluate_saved_model(model_path, dataset, batch_size=64):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the entire model
    model = torch.load(model_path)
    model = model.to(device)

    # Create data loader using the same dataset
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # Evaluate model
    metrics_df, overall_metrics = evaluate_model(model, data_loader, device)

    # Print results
    print_evaluation_results(metrics_df, overall_metrics)

    return metrics_df, overall_metrics

# Define the SpectrogramDataset
class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.inputs = torch.FloatTensor(dataframe.iloc[:, :513].values)
        self.labels = torch.FloatTensor(dataframe.iloc[:, 513:].values)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = self.inputs[idx]
        label_tensor = self.labels[idx]
        input_tensor = input_tensor.view(1, 513)
        return input_tensor, label_tensor

# Main training and evaluation loop
def main():
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # List of your large parquet file paths (assuming you have 3 large files)
    file_list = [f'data/large_datafile_{i}.parquet' for i in range(1, 4)]  # 3 files: large_datafile_1.parquet, etc.

    # Initialize the model and move it to the device
    model = PitchDetectionCRNN(num_pitches=88).to(device)

    # Define loss function and optimizer
    estimated_negative_count = 0.95  # Adjust based on your data knowledge
    estimated_positive_count = 0.05
    pos_weight = torch.full((88,), estimated_negative_count / estimated_positive_count).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)

    # Prepare a separate validation dataset
    validation_dataframes = []

    # First pass: Collect validation data from each file
    for file in file_list:
        print(f"Loading {file} for validation data extraction")
        df = pd.read_parquet(file)
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split a portion for validation (e.g., 10% from each file)
        val_size = int(len(df) * 0.1)
        val_data = df[:val_size]
        validation_dataframes.append(val_data)

        # Save the remaining data back to the file for training
        train_data = df[val_size:]
        # Optionally, overwrite the file or save to a new one
        train_file = file.replace('.parquet', '_train.parquet')
        train_data.to_parquet(train_file, index=False)
        print(f"Saved training data to {train_file}")

    # Concatenate validation data
    val_data = pd.concat(validation_dataframes, ignore_index=True)
    print(f"Total validation samples: {len(val_data)}")

    # Create validation dataset and loader
    val_dataset = SpectrogramDataset(val_data)
    batch_size = 64  # Adjust based on memory constraints
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training loop with early stopping
    num_epochs = 30  # Adjust as needed
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait before early stopping
    counter = 0

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # For each training data file
            for file_index, file in enumerate(file_list):
                train_file = file.replace('.parquet', '_train.parquet')
                print(f"Loading {train_file} for training")
                train_df = pd.read_parquet(train_file)

                # Create training dataset and loader
                train_dataset = SpectrogramDataset(train_df)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Train on current training data
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                print(f"Training Loss on file {file_index+1}/{len(file_list)}: {train_loss:.4f}")

            # Validate after each epoch
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model, 'best_pitch_model.pth')
                print("Saved best model!")
            else:
                counter += 1
                print(f"No improvement in validation loss for {counter} epochs.")
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Save the final model
        torch.save(model, 'final_pitch_model.pth')
        print("Training complete. Final model saved.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        torch.save(model.state_dict(), 'model_error_state.pth')
        raise

    # Evaluate the saved model
    print("\nEvaluating the best saved model on validation data...")
    metrics_df, overall_metrics = evaluate_saved_model(
        model_path='best_pitch_model.pth',
        dataset=val_dataset,
        batch_size=batch_size
    )

    # Optionally, save the metrics to a CSV file
    metrics_df.to_csv('pitch_detection_metrics.csv', index=False)
    print("Evaluation metrics saved to 'pitch_detection_metrics.csv'.")

if __name__ == "__main__":
    main()
    # Wait for 60 seconds before shutting down
    print("Training completed. The system will shut down in 60 seconds.")
    time.sleep(60)
    os.system("sudo shutdown -h now")
