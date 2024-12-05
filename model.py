import os
import time
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

############################################################
# Custom IterableDataset
#
# Changes:
# - Removed positive sampling logic.
# - Instead of yielding row-by-row, we yield chunks of 240 rows.
# - For each 240-row chunk, the input is [0:240], and the target is the middle [80:160] rows.
# - We now read the entire file at once (for simplicity) and then create windows.
#
# We assume each file has been padded so that its total length N is divisible by 80.
# Thus, we can form sliding windows:
# Window 1: Input rows [0:240), target rows [80:160)
# Window 2: Input rows [80:320), target rows [160:240)
# and so forth, stepping by 80 each time until we can no longer form a full 240-window.
#
# After forming each window, we reshape:
# Input: (1, 1, 240, 513)
# Target: (1, 80, 88)
#
# No sampling probability is applied now; we yield every window.
############################################################

class SpectrogramIterableDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        for file_path in self.file_list:
            yield from self.parse_file(file_path)

    def parse_file(self, file_path):
        # Read entire file into a single DataFrame
        parquet_file = pq.ParquetFile(file_path)
        df = parquet_file.read().to_pandas()

        # Split into inputs (first 513 columns) and labels (last 88 columns)
        inputs = torch.FloatTensor(df.iloc[:, :513].values)   # shape: (N, 513)
        labels = torch.FloatTensor(df.iloc[:, 513:].values)   # shape: (N, 88)

        total_rows = inputs.shape[0]

        # We will form windows of length 240 rows, stepping by 80 rows
        # Each window predicts the middle 80 rows.
        # Window start indices: 0, 80, 160, ... until start+240 <= total_rows
        step = 80
        window_size = 240
        mid_size = 80  # The number of rows in the middle we want to predict

        for start in range(0, total_rows - window_size + 1, step):
            end = start + window_size
            # Input chunk: [start : start+240]
            input_chunk = inputs[start:end, :]  # (240, 513)

            # Label chunk: [start+80 : start+160] for the middle 80 rows
            label_start = start + 80
            label_end = label_start + mid_size
            label_chunk = labels[label_start:label_end, :]  # (80, 88)

            # Reshape input:
            # The model expects shape [batch, channel, height, width]
            # height=240 (time), width=513 (freq), channel=1, batch=1
            input_tensor = input_chunk.view(1, 1, 240, 513)

            # Labels: (1, 80, 88)
            label_tensor = label_chunk.view(1, mid_size, 88)

            yield input_tensor, label_tensor

def get_data_loader(dataset, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size)
    return data_loader

############################################################
# Model Definition
#
# Changes:
# - Previously, the model output was (batch, 88) for a single row prediction.
# - Now, we must predict 80 rows of 88 pitches.
# - Final layer outputs 80 * 88 = 7040 units, then we reshape to (batch, 80, 88).
#
# Input shape: (batch, 1, 240, 513)
# After convolutions and pooling over frequency dimension:
# The final Flattened size must be reconsidered.
# Let's re-check the dimension after conv and pooling:
#
# Convs with kernel=(1,3), pooling=(1,2) reduces width dimension by factor 4:
# - Original width = 513
# - After first MaxPool2d((1,2)): width ~ 513/2 = 256.5 -> floor to 256
# - After second MaxPool2d((1,2)): width ~ 256/2 = 128
#
# Height remains 240 because kernel=(1,3) acts only on width dimension, and no pooling on height dimension.
#
# So final layer input size = channels * height * width = 128 * 240 * 128?
#
# Wait, this is different from original code. The original code snippet had comments that were for single-row input.
#
# We must carefully reconsider:
#
# conv_layers:
# 1) Input: (1, 240, 513)
#    Conv2d(1->32, kernel=(1,3), padding=(0,1)) => width stays the same (513),
#    BatchNorm2d(32), ReLU,
#    MaxPool2d((1,2)) => width halved: 513/2 ~256. So after first block: (32, 240, 256)
#
# 2) Conv2d(32->64, (1,3), padding=(0,1)) => width still 256,
#    BatchNorm2d(64), ReLU,
#    MaxPool2d((1,2)) => width halved again: 256/2=128. after second block: (64,240,128)
#
# 3) Conv2d(64->128, (1,3), padding=(0,1)) => width still 128,
#    BatchNorm2d(128), ReLU,
#    no pooling here, final shape: (128, 240, 128)
#
# Flatten: 128 * 240 * 128 = 128 * 30720 = 3,932,160 features -> huge number
# This is extremely large. We must consider if we need to pool over the time dimension as well or reduce complexity. The original code was for a single row. With 240 rows, we have a massive input.
#
# If we keep it as is, the model is enormous. Let's trust that the user wants minimal changes:
#
# We'll produce a huge linear layer. This is probably not practical. The user must realize the model is huge.
#
# Let's proceed anyway. We'll do:
# fc_layers:
#    input: 128 * 240 * 128 = 3,932,160
#    linear -> 256 -> linear -> 7040 (80*88)
#
# It's huge but let's follow instructions.
#
# We'll produce final predictions and reshape: x = x.view(batch_size, 80, 88).
############################################################

class PitchDetectionModel(nn.Module):
    def __init__(self, num_pitches=88, mid_size=80):
        super(PitchDetectionModel, self).__init__()
        self.num_pitches = num_pitches
        self.mid_size = mid_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # After these layers:
        # Input: (batch,1,240,513)
        # After first pool: (batch,32,240,256)
        # After second pool: (batch,64,240,128)
        # After third conv (no pool): (batch,128,240,128)
        flattened_size = 128 * 240 * 128

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.mid_size * self.num_pitches),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        # Reshape to (batch, 80, 88)
        batch_size = x.size(0)
        x = x.view(batch_size, self.mid_size, self.num_pitches)
        return x

############################################################
# Training, Validation, and Evaluation:
#
# Changes:
# - The output and target now have shape (batch,80,88).
# - The criterion (BCE) can handle this shape directly.
# - For evaluation, we flatten the (80) dimension along with batch to treat them as multiple samples.
############################################################

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)  # output shape: (batch,80,88)
        loss = criterion(output, target)  # target shape: (batch,80,88)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

    return total_loss / num_batches

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # (batch,80,88)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

def evaluate_model(model, data_loader, device, threshold=0.5):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)  # (batch,80,88), Sigmoid applied inside model
            predictions = (outputs > threshold).float()

            # Reshape for metrics:
            # Flatten batch and time (80) dimension
            batch_size = predictions.size(0)
            # shape: (batch*80, 88)
            preds_2d = predictions.view(batch_size * 80, 88).cpu().numpy()
            targs_2d = target.view(batch_size * 80, 88).cpu().numpy()

            all_predictions.append(preds_2d)
            all_targets.append(targs_2d)

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, zero_division=0
    )
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_targets.flatten(), all_predictions.flatten(), average='binary', zero_division=0
    )

    pitch_accuracy = (all_targets == all_predictions).mean(axis=0)
    overall_accuracy = accuracy_score(all_targets.flatten(), all_predictions.flatten())

    metrics_df = pd.DataFrame({
        'Pitch': [f'Pitch_{i}' for i in range(len(precision))],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': pitch_accuracy
    })

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

    print("\n=== Performance Distribution ===")
    print("\nF1 Score Distribution:")
    print(metrics_df['F1 Score'].describe())

def evaluate_saved_model(model_path, dataset, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model = model.to(device)

    data_loader = get_data_loader(dataset, batch_size=batch_size)
    metrics_df, overall_metrics = evaluate_model(model, data_loader, device)

    print_evaluation_results(metrics_df, overall_metrics)
    return metrics_df, overall_metrics

############################################################
# Main function
#
# Changes:
# - We assume training and testing (validation) data are already separated outside.
# - The dataset classes changed from WeightedSpectrogramIterableDataset to SpectrogramIterableDataset as we removed sampling.
#
# Update the file paths and dataset accordingly.
############################################################

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Training and validation files (adjust as needed)
    train_files = [f'data/train_batch_{i}.parquet' for i in range(1, 11)]
    val_files = ['data/test_dataframe.parquet']  # single file or list

    # Create datasets
    train_dataset = SpectrogramIterableDataset(train_files)
    val_dataset = SpectrogramIterableDataset(val_files)

    batch_size = 4  # Since each sample is quite large, use a small batch size
    train_loader = get_data_loader(train_dataset, batch_size=batch_size)
    val_loader = get_data_loader(val_dataset, batch_size=batch_size)

    model = PitchDetectionModel(num_pitches=88).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    num_epochs = 20
    best_val_loss = float('inf')
    patience = 15
    counter = 0

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)

            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")

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

        torch.save(model, 'final_pitch_model.pth')
        print("Training complete. Final model saved.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        torch.save(model.state_dict(), 'model_error_state.pth')
        raise

    print("\nEvaluating the best saved model on validation data...")
    metrics_df, overall_metrics = evaluate_saved_model(
        model_path='best_pitch_model.pth',
        dataset=val_dataset,
        batch_size=batch_size
    )

    metrics_df.to_csv('pitch_detection_metrics.csv', index=False)
    print("Evaluation metrics saved to 'pitch_detection_metrics.csv'.")

if __name__ == "__main__":
    main()
    print("Training completed. The system will shut down in 60 seconds.")
    time.sleep(60)
    os.system("sudo shutdown -h now")
