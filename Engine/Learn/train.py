## Training functions and scripts for TCN and LSTM models, including binary classification setup and review notebooks.

import pandas as pd
import numpy as np
import math
import torch
from tqdm import tqdm

from Loaders import SequenceDataset
from torch.utils.data import DataLoader
from Loss import BinaryTradeProfitabilityLoss

def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine


def train_binary_models(
    df: pd.DataFrame,
    binary_task: str,
    features_function,
    preprocess_function,
    model,
    model_params: dict,
    training_params: dict,
    criterion_param: dict,
    split=True
):
    """
    Train separate models for BUY and SELL signals based on the specified binary task.
    
    Args:
        df (pd.DataFrame): Input dataframe containing features and target labels.
        binary_task (str): 'buy' or 'sell' to specify which model to train.
        model_params (dict): Dictionary of model hyperparameters.
        training_params (dict): Dictionary of training hyperparameters.
    
    Returns:
        Model pack containing the trained model, model parameters, and training history.
    """

    # Configure binary task and derive labels from outcomes
    if binary_task == 'buy':
        df['y'] = df['target'].replace(-1, 0)  # BUY model: positive = take BUY trade
    elif binary_task == 'sell':
        df['y'] = df['target'].replace(1, 0).replace(-1, 1)  # SELL model: positive = take SELL trade
    else:
        raise ValueError("binary_task must be either 'buy' or 'sell'")

    # Generate features and preprocess data
    X = features_function(df)
    X, y = preprocess_function(X, df['y'])

    # Split into train and test sets if requested
    if split:
        test_date = '2025-12-25'
        X_train = X[df['Time'] < test_date]
        y_train = y[df['Time'] < test_date]
        X_test = X[df['Time'] >= test_date]
        y_test = y[df['Time'] >= test_date]
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y   

    # Create PyTorch datasets and dataloaders

    seq_len = training_params.get('seq_len', 256)  # default sequence length if not specified
    batch_size = training_params.get('batch_size', 256)  # default batch size if not specified

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets with trade_outcomes
    train_ds = SequenceDataset(X_train, y_train, seq_len=seq_len, df_idx=None, 
                               custom_targets=None, trade_outcomes=None)
    
    val_ds = SequenceDataset(X_test, y_test, seq_len=seq_len, df_idx=None, 
                             custom_targets=None, trade_outcomes=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)


    # Initialize training paramaters
    
    base_lr = 1e-4
    weight_decay = 1e-3

    counts = np.bincount(y_train)
    counts = np.maximum(counts, 1)
    weights = (len(y_train) / (len(counts) * counts)).astype(float)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    criterion = BinaryTradeProfitabilityLoss(alpha=class_weights, **criterion_param)

    # Initialize and train model
    model = model(**model_params).to(device)

    n_epochs = training_params.get('n_epochs', 10)  # default number of epochs if not specified
    warmup_steps = training_params.get('warmup_steps', 200)  # default warmup epochs if not specified
    total_steps = len(train_loader) * n_epochs

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_steps, total_steps))
    global_step = 0

    ## Start training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch} Training Progress"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * X_batch.size(0)
            global_step += 1

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_epoch_loss:.4f}")

    # After training, evaluate on validation set
    if split:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, _ in tqdm(val_loader, desc="Validation Progress"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_val_loss:.4f}")