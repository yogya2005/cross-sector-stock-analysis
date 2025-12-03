"""
Utility functions for data loading and general helper operations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load preprocessed stock market data.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(filepath)
    return df


def prepare_features_targets(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Split dataframe into features (PCs) and targets (sector labels).
    
    Args:
        df: Input dataframe with PC features and sector targets
        
    Returns:
        Tuple of (X: feature matrix, y_targets: target dataframe, sector_names: list of sector column names)
    """
    # Identify PC columns and target columns
    pc_cols = [col for col in df.columns if col.startswith('PC')]
    target_cols = [col for col in df.columns if 'Target' in col]
    
    X = df[pc_cols].values
    y_targets = df[target_cols]
    
    return X, y_targets, target_cols


def save_figure(fig, filename: str, results_dir: str = "results/figures"):
    """
    Save matplotlib figure to results directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of file to save
        results_dir: Directory to save to
    """
    import os
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")


def set_plot_style():
    """Set consistent plotting style for all visualizations."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
