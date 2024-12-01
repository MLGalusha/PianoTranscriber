import os
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Custom IterableDataset with Weighted Sampling
class WeightedSpectrogramIterableDataset(IterableDataset):
    def __init__(self, file_list, positive_sample_rate=0.5):
        self.file_list = file_list
        self.positive_sample_rate = positive_sample_rate  # Desired rate of positive samples

    def __iter__(self):
        for file_path in self.file_list:
            yield from self.parse_file(file_path)

    def parse_file(self, file_path):
        parquet_file = pq.ParquetFile(file_path)
        batch_size = 5000  # Adjust as needed

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_df = batch.to_pandas()

            inputs = torch.FloatTensor(batch_df.iloc[:, :513].values)
            labels = torch.FloatTensor(batch_df.iloc[:, 513:].values)

            for input_tensor, label_tensor in zip(inputs, labels):
                # Corrected reshaping
                input_tensor = input_tensor.view(1, 1, 513)  # [channels, height, width]

                # Check if the sample is positive (has at least one active pitch)
                is_positive = label_tensor.sum() > 0

                # Apply sampling probability
                if is_positive:
                    # Include positive samples
                    yield input_tensor, label_tensor
                else:
                    # Include negative samples with adjusted probability
                    if random.random() < (1 - self.positive_sample_rate):
                        yield input_tensor, label_tensor

# Function to create DataLoader
def get_data_loader(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader

# Define the CNN model
class PitchDetectionModel(nn.Module):
    def __init__(self, num_pitches=88):
        super(PitchDetectionModel, self).__init__()

        # Reduced number of pooling layers and smaller kernels
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # Only pool frequency dimension

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            # Third conv block without pooling
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate flattened feature size
        # Input dimensions: [batch_size, channels=1, height=1, width=513]
        # After conv and pooling:
        # Height remains 1
        # Width after pooling: 513 / (2 * 2) = 128.25 -> floor to 128
        # Channels after last conv layer: 128
        self.flattened_size = 128 * 1 * 128  # channels * height * width

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten all dimensions except batch
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_pitches),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

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

            # Since the model outputs probabilities due to Sigmoid, no need to apply Sigmoid again
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
    data_loader = get_data_loader(dataset, batch_size=batch_size)

    # Evaluate model
    metrics_df, overall_metrics = evaluate_model(model, data_loader, device)

    # Print results
    print_evaluation_results(metrics_df, overall_metrics)

    return metrics_df, overall_metrics

# Main training and evaluation loop
def main():
    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # List of your batch file paths
    file_list = [f'data/master_dataframe_batch_{i}.parquet' for i in range(1, 3)]  # Adjust file indices as needed

    # Shuffle the file list to ensure randomness
    random.shuffle(file_list)

    # Split file list into training and validation sets
    split_ratio = 0.9  # 90% training, 10% validation
    split_index = int(len(file_list) * split_ratio)
    train_files = file_list[:split_index]
    val_files = file_list[split_index:]

    # Create datasets with WeightedSampling
    train_dataset = WeightedSpectrogramIterableDataset(train_files, positive_sample_rate=0.5)
    val_dataset = WeightedSpectrogramIterableDataset(val_files, positive_sample_rate=1.0)  # Use full data for validation

    # Create data loaders
    batch_size = 64  # Adjust based on memory constraints
    train_loader = get_data_loader(train_dataset, batch_size=batch_size)
    val_loader = get_data_loader(val_dataset, batch_size=batch_size)

    # Initialize the model and move it to the device
    model = PitchDetectionModel(num_pitches=88).to(device)

    # Define loss function and optimizer
    # Since the model outputs probabilities (Sigmoid activation), use BCELoss
    # If you want to use pos_weight, you need to use BCEWithLogitsLoss and remove Sigmoid from the model
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Training loop with early stopping
    num_epochs = 20  # Adjust as needed
    best_val_loss = float('inf')
    patience = 15  # Number of epochs to wait before early stopping
    counter = 0

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)

            print(f"Training Loss: {train_loss:.4f}")
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

