import os
import random
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Define the IterableDataset
class SpectrogramIterableDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def parse_file(self, file_path):
        # Open the Parquet file using PyArrow
        parquet_file = pq.ParquetFile(file_path)

        # Define batch size for reading chunks
        batch_size = 5000  # Adjust based on memory constraints

        # Iterate over the file in batches
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_df = batch.to_pandas()

            # Split into inputs and labels
            inputs_df = batch_df.iloc[:, :513]
            labels_df = batch_df.iloc[:, 513:]

            # Convert to tensors
            inputs = torch.FloatTensor(inputs_df.values)
            labels = torch.FloatTensor(labels_df.values)

            # Yield individual samples
            for input_tensor, label_tensor in zip(inputs, labels):
                # Reshape input_tensor to match model's expected input shape
                input_tensor = input_tensor.view(1, 1, 513)  # Shape: [channels, time, freq]
                yield input_tensor, label_tensor

    def __iter__(self):
        for file_path in self.file_list:
            yield from self.parse_file(file_path)

# Function to create DataLoader
def get_data_loader(file_list, batch_size):
    dataset = SpectrogramIterableDataset(file_list)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader

# Define the neural network model
class PitchDetectionModel(nn.Module):
    def __init__(self, num_pitches=88):
        super(PitchDetectionModel, self).__init__()

        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # Pooling over frequency dimension

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate the size of the flattened features
        # Input frequency dimension reduces from 513 to 128 after pooling
        # So the feature map size is [batch_size, 128, 1, 128]
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten all dimensions except batch
            nn.Linear(128 * 1 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_pitches),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Training function
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

    average_loss = total_loss / len(data_loader)
    return average_loss

# Validation function
def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
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

def evaluate_saved_model(model_path, file_list, batch_size=128):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the entire model
    model = torch.load(model_path)
    model = model.to(device)

    # Create data loader using the same dataset class
    data_loader = get_data_loader(file_list, batch_size=batch_size)

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
    file_list = [f'data/master_dataframe_batch_{i}.parquet' for i in range(1, 5)]

    # Shuffle the file list to ensure randomness
    random.shuffle(file_list)

    # Split file list into training and validation sets
    split_ratio = 0.8  # 80% training, 20% validation
    split_index = int(len(file_list) * split_ratio)
    train_files = file_list[:split_index]
    val_files = file_list[split_index:]

    # Create data loaders
    batch_size = 128  # Adjust based on memory constraints
    train_loader = get_data_loader(train_files, batch_size=batch_size)
    val_loader = get_data_loader(val_files, batch_size=batch_size)

    # Initialize the model and move it to the device
    model = PitchDetectionModel(num_pitches=88)  # 88 pitches
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20  # Adjust number of epochs
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the entire model
            torch.save(model, 'best_pitch_model.pth')
            print("Saved best model!")

    # Save the final model
    torch.save(model, 'final_pitch_model.pth')
    print("Training complete. Final model saved.")

    # Evaluate the saved model
    print("\nEvaluating the best saved model on validation data...")
    metrics_df, overall_metrics = evaluate_saved_model(
        model_path='best_pitch_model.pth',
        file_list=val_files,
        batch_size=batch_size
    )

    # Optionally, save the metrics to a CSV file
    metrics_df.to_csv('pitch_detection_metrics.csv', index=False)
    print("Evaluation metrics saved to 'pitch_detection_metrics.csv'.")

# Run the main function
if __name__ == "__main__":
    main()