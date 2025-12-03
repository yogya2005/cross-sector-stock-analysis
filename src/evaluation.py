"""
Evaluation and cross-sector analysis functions.

This module contains functions for analyzing sector performance by market regime,
creating visualizations, and generating insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def analyze_sector_by_regime(df: pd.DataFrame, cluster_labels: np.ndarray,
                             sector_target_cols: List[str],
                             cluster_names: Optional[Dict[int, str]] = None) -> Dict:
    """
    Analyze sector performance for each market regime (cluster).
    
    Args:
        df: Original dataframe with sector targets
        cluster_labels: Cluster assignments for each sample
        sector_target_cols: List of sector target column names
        cluster_names: Optional mapping of cluster IDs to names
        
    Returns:
        Dictionary with sector performance analysis
    """
    # Add cluster labels to dataframe
    df_analysis = df.copy()
    df_analysis['Cluster'] = cluster_labels
    
    # Calculate sector win rates by cluster
    results = {}
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
        
        sector_performance = {}
        for sector_col in sector_target_cols:
            # Calculate percentage of days sector went up
            win_rate = cluster_data[sector_col].mean() * 100
            count = len(cluster_data)
            sector_performance[sector_col.replace('_Target', '')] = {
                'win_rate': win_rate,
                'count': count
            }
        
        cluster_name = cluster_names.get(cluster, f'Cluster {cluster}') if cluster_names else f'Cluster {cluster}'
        results[cluster_name] = sector_performance
    
    return results


def create_sector_regime_heatmap(sector_performance: Dict,
                                 title: str = 'Sector Performance by Market Regime',
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap showing sector performance across different market regimes.
    
    Args:
        sector_performance: Dictionary from analyze_sector_by_regime
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Convert to DataFrame for heatmap
    data = {}
    for regime, sectors in sector_performance.items():
        data[regime] = {sector: info['win_rate'] for sector, info in sectors.items()}
    
    df_heatmap = pd.DataFrame(data).T
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                vmin=0, vmax=100, cbar_kws={'label': '% Days Sector Up'},
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Sector', fontsize=12)
    ax.set_ylabel('Market Regime', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def identify_sector_correlations(df: pd.DataFrame, sector_target_cols: List[str],
                                 cluster_labels: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate sector correlations overall and by cluster.
    
    Args:
        df: Dataframe with sector targets
        sector_target_cols: List of sector target column names
        cluster_labels: Optional cluster labels to analyze by regime
        
    Returns:
        Dictionary with correlation matrices
    """
    results = {}
    
    # Overall correlation
    sector_data = df[sector_target_cols]
    sector_names = [col.replace('_Target', '') for col in sector_target_cols]
    corr_overall = sector_data.corr()
    corr_overall.index = sector_names
    corr_overall.columns = sector_names
    results['overall'] = corr_overall
    
    # By cluster if provided
    if cluster_labels is not None:
        df_temp = df.copy()
        df_temp['Cluster'] = cluster_labels
        
        for cluster in np.unique(cluster_labels):
            cluster_data = df_temp[df_temp['Cluster'] == cluster][sector_target_cols]
            corr_cluster = cluster_data.corr()
            corr_cluster.index = sector_names
            corr_cluster.columns = sector_names
            results[f'Cluster {cluster}'] = corr_cluster
    
    return results


def plot_sector_correlations(correlation_matrix: pd.DataFrame,
                             title: str = 'Sector Correlations',
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot sector correlation heatmap.
    
    Args:
        correlation_matrix: Correlation matrix
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={'label': 'Correlation'}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def identify_regime_characteristics(X: np.ndarray, cluster_labels: np.ndarray,
                                    feature_names: List[str],
                                    top_n: int = 10) -> Dict:
    """
    Identify characteristics of each market regime based on feature values.
    
    Args:
        X: Feature matrix
        cluster_labels: Cluster assignments
        feature_names: List of feature names
        top_n: Number of top features to show per cluster
        
    Returns:
        Dictionary with regime characteristics
    """
    results = {}
    
    for cluster in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster
        cluster_data = X[cluster_mask]
        
        # Calculate mean feature values for this cluster
        mean_values = np.mean(cluster_data, axis=0)
        
        # Calculate deviation from overall mean
        overall_mean = np.mean(X, axis=0)
        deviations = mean_values - overall_mean
        
        # Get top features by absolute deviation
        top_indices = np.argsort(np.abs(deviations))[-top_n:][::-1]
        
        characteristics = []
        for idx in top_indices:
            characteristics.append({
                'feature': feature_names[idx],
                'cluster_mean': mean_values[idx],
                'overall_mean': overall_mean[idx],
                'deviation': deviations[idx]
            })
        
        results[f'Cluster {cluster}'] = {
            'n_samples': np.sum(cluster_mask),
            'top_characteristics': pd.DataFrame(characteristics)
        }
    
    return results


def generate_insights_summary(sector_performance: Dict,
                              regime_characteristics: Dict) -> str:
    """
    Generate text summary of cross-sector insights.
    
    Args:
        sector_performance: Results from analyze_sector_by_regime
        regime_characteristics: Results from identify_regime_characteristics
        
    Returns:
        String with formatted insights
    """
    insights = []
    insights.append("=== CROSS-SECTOR ANALYSIS INSIGHTS ===\n")
    
    for regime, sectors in sector_performance.items():
        insights.append(f"\n{regime}:")
        insights.append(f"  Sample size: {list(sectors.values())[0]['count']} days")
        
        # Find best and worst performing sectors
        sector_rates = {sector: info['win_rate'] for sector, info in sectors.items()}
        best_sectors = sorted(sector_rates.items(), key=lambda x: x[1], reverse=True)[:3]
        worst_sectors = sorted(sector_rates.items(), key=lambda x: x[1])[:3]
        
        insights.append("  Top performing sectors:")
        for sector, rate in best_sectors:
            insights.append(f"    - {sector}: {rate:.1f}% win rate")
        
        insights.append("  Weakest performing sectors:")
        for sector, rate in worst_sectors:
            insights.append(f"    - {sector}: {rate:.1f}% win rate")
    
    return '\n'.join(insights)


def plot_regime_distribution(cluster_labels: np.ndarray,
                             cluster_names: Optional[Dict[int, str]] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of samples across market regimes.
    
    Args:
        cluster_labels: Cluster assignments
        cluster_names: Optional mapping of cluster IDs to names
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    labels = [cluster_names.get(c, f'Cluster {c}') if cluster_names else f'Cluster {c}'
              for c in unique]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
    bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add percentage labels
    total = len(cluster_labels)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Number of Days', fontsize=12)
    ax.set_xlabel('Market Regime', fontsize=12)
    ax.set_title('Distribution of Market Regimes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_results_to_csv(sector_performance: Dict, output_path: str):
    """
    Save sector performance results to CSV.
    
    Args:
        sector_performance: Results from analyze_sector_by_regime
        output_path: Path to save CSV file
    """
    # Flatten the nested dictionary
    rows = []
    for regime, sectors in sector_performance.items():
        for sector, info in sectors.items():
            rows.append({
                'Market_Regime': regime,
                'Sector': sector,
                'Win_Rate': info['win_rate'],
                'Sample_Size': info['count']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
