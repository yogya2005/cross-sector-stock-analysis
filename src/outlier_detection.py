"""
Outlier detection methods for identifying anomalous market days.

This module implements Isolation Forest and Local Outlier Factor (LOF)
to detect unusual market conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def detect_outliers_isolation_forest(X: np.ndarray, contamination: float = 0.05,
                                     random_state: int = 42) -> Dict:
    """
    Detect outliers using Isolation Forest.
    
    Args:
        X: Feature matrix
        contamination: Expected proportion of outliers
        random_state: Random seed
        
    Returns:
        Dictionary with model, predictions, scores, and outlier indices
    """
    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(X)
    scores = model.score_samples(X)
    
    # -1 for outliers, 1 for inliers
    outlier_mask = predictions == -1
    outlier_indices = np.where(outlier_mask)[0]
    
    return {
        'model': model,
        'predictions': predictions,
        'scores': scores,
        'outlier_mask': outlier_mask,
        'outlier_indices': outlier_indices,
        'n_outliers': np.sum(outlier_mask),
        'outlier_percentage': np.mean(outlier_mask) * 100
    }


def detect_outliers_lof(X: np.ndarray, n_neighbors: int = 20,
                       contamination: float = 0.05) -> Dict:
    """
    Detect outliers using Local Outlier Factor.
    
    Args:
        X: Feature matrix
        n_neighbors: Number of neighbors to consider
        contamination: Expected proportion of outliers
        
    Returns:
        Dictionary with model, predictions, scores, and outlier indices
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = model.fit_predict(X)
    scores = model.negative_outlier_factor_
    
    # -1 for outliers, 1 for inliers
    outlier_mask = predictions == -1
    outlier_indices = np.where(outlier_mask)[0]
    
    return {
        'model': model,
        'predictions': predictions,
        'scores': scores,
        'outlier_mask': outlier_mask,
        'outlier_indices': outlier_indices,
        'n_outliers': np.sum(outlier_mask),
        'outlier_percentage': np.mean(outlier_mask) * 100
    }


def visualize_outliers_2d(X: np.ndarray, outlier_mask: np.ndarray, 
                         method_name: str = 'Isolation Forest',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize outliers in 2D using PCA.
    
    Args:
        X: Feature matrix
        outlier_mask: Boolean mask where True indicates outlier
        method_name: Name of outlier detection method
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot inliers
    inlier_mask = ~outlier_mask
    ax.scatter(X_2d[inlier_mask, 0], X_2d[inlier_mask, 1],
              c='blue', label='Inliers', alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    
    # Plot outliers
    ax.scatter(X_2d[outlier_mask, 0], X_2d[outlier_mask, 1],
              c='red', label='Outliers', alpha=0.8, s=80, marker='X',
              edgecolors='darkred', linewidths=1.5)
    
    explained_var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'Outlier Detection: {method_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add text with outlier percentage
    n_outliers = np.sum(outlier_mask)
    outlier_pct = np.mean(outlier_mask) * 100
    ax.text(0.02, 0.98, f'Outliers: {n_outliers} ({outlier_pct:.1f}%)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_outlier_methods(if_results: Dict, lof_results: Dict) -> Dict:
    """
    Compare Isolation Forest and LOF outlier detection results.
    
    Args:
        if_results: Results from Isolation Forest
        lof_results: Results from LOF
        
    Returns:
        Dictionary with comparison statistics
    """
    if_outliers = set(if_results['outlier_indices'])
    lof_outliers = set(lof_results['outlier_indices'])
    
    # Overlap analysis
    overlap = if_outliers & lof_outliers
    only_if = if_outliers - lof_outliers
    only_lof = lof_outliers - if_outliers
    union = if_outliers | lof_outliers
    
    overlap_pct = len(overlap) / len(union) * 100 if len(union) > 0 else 0
    
    return {
        'if_outliers': if_outliers,
        'lof_outliers': lof_outliers,
        'overlap': overlap,
        'only_if': only_if,
        'only_lof': only_lof,
        'n_overlap': len(overlap),
        'n_only_if': len(only_if),
        'n_only_lof': len(only_lof),
        'overlap_percentage': overlap_pct
    }


def analyze_outlier_dates(df: pd.DataFrame, outlier_indices: np.ndarray,
                         date_column: str = 'Date') -> pd.DataFrame:
    """
    Analyze dates of detected outliers to identify significant events.
    
    Args:
        df: Original dataframe with Date column
        outlier_indices: Indices of outlier samples
        date_column: Name of date column
        
    Returns:
        DataFrame with outlier dates and basic statistics
    """
    outlier_df = df.iloc[outlier_indices].copy()
    
    if date_column in outlier_df.columns:
        outlier_df[date_column] = pd.to_datetime(outlier_df[date_column])
        outlier_df = outlier_df.sort_values(date_column)
    
    return outlier_df


def plot_outlier_comparison(if_results: Dict, lof_results: Dict,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create Venn diagram-style comparison of outlier detection methods.
    
    Args:
        if_results: Results from Isolation Forest
        lof_results: Results from LOF
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    comparison = compare_outlier_methods(if_results, lof_results)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Both Methods', 'Only IF', 'Only LOF']
    counts = [comparison['n_overlap'], comparison['n_only_if'], comparison['n_only_lof']]
    colors = ['purple', 'red', 'orange']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Outliers', fontsize=12)
    ax.set_title('Outlier Detection Method Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add overlap percentage
    overlap_text = f'Overlap: {comparison["overlap_percentage"]:.1f}%'
    ax.text(0.5, 0.95, overlap_text, transform=ax.transAxes,
            fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
