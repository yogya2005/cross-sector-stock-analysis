"""
Clustering algorithms for identifying market regimes.

This module implements K-Means and Hierarchical clustering to discover
distinct market conditions in stock market data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def kmeans_analysis(X: np.ndarray, k_range: Tuple[int, int] = (3, 10), 
                    random_state: int = 42) -> Dict:
    """
    Perform K-Means clustering analysis with multiple k values.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        k_range: Tuple of (min_k, max_k) to test
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - 'metrics': DataFrame with metrics for each k
            - 'optimal_k': Best k value
            - 'model': Trained KMeans model with optimal k
            - 'labels': Cluster labels for optimal k
            - 'inertias': List of inertias for elbow plot
    """
    results = {
        'k_values': [],
        'inertias': [],
        'silhouette_scores': [],
        'calinski_scores': [],
        'davies_bouldin_scores': []
    }
    
    models = {}
    
    # Test different k values
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        results['k_values'].append(k)
        results['inertias'].append(kmeans.inertia_)
        results['silhouette_scores'].append(silhouette_score(X, labels))
        results['calinski_scores'].append(calinski_harabasz_score(X, labels))
        results['davies_bouldin_scores'].append(davies_bouldin_score(X, labels))
        
        models[k] = kmeans
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'k': results['k_values'],
        'inertia': results['inertias'],
        'silhouette': results['silhouette_scores'],
        'calinski_harabasz': results['calinski_scores'],
        'davies_bouldin': results['davies_bouldin_scores']
    })
    
    # Select optimal k based on silhouette score
    optimal_k = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
    
    return {
        'metrics': metrics_df,
        'optimal_k': int(optimal_k),
        'model': models[int(optimal_k)],
        'labels': models[int(optimal_k)].labels_,
        'inertias': results['inertias']
    }


def plot_elbow_silhouette(metrics_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot elbow method and silhouette scores.
    
    Args:
        metrics_df: DataFrame with clustering metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    axes[0].plot(metrics_df['k'], metrics_df['inertia'], marker='o', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette plot
    axes[1].plot(metrics_df['k'], metrics_df['silhouette'], marker='o', 
                 color='green', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score by K', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Mark optimal k
    optimal_idx = metrics_df['silhouette'].idxmax()
    optimal_k = metrics_df.loc[optimal_idx, 'k']
    axes[1].scatter(optimal_k, metrics_df.loc[optimal_idx, 'silhouette'], 
                    color='red', s=200, zorder=5, label=f'Optimal k={int(optimal_k)}')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_clusters_2d(X: np.ndarray, labels: np.ndarray, method: str = 'pca',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize clusters in 2D using PCA or t-SNE.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        method: 'pca' or 'tsne'
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        title = f'K-Means Clustering Visualization (PCA)'
        explained_var = reducer.explained_variance_ratio_
        xlabel = f'PC1 ({explained_var[0]:.1%} variance)'
        ylabel = f'PC2 ({explained_var[1]:.1%} variance)'
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = reducer.fit_transform(X)
        title = f'K-Means Clustering Visualization (t-SNE)'
        xlabel = 't-SNE Component 1'
        ylabel = 't-SNE Component 2'
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=[color], label=f'Cluster {label}',
                  alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def hierarchical_clustering(X: np.ndarray, n_clusters: Optional[int] = None,
                           method: str = 'ward', random_state: int = 42) -> Dict:
    """
    Perform hierarchical clustering.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters (if None, will be determined from dendrogram)
        method: Linkage method ('ward', 'complete', 'average')
        random_state: Random seed
        
    Returns:
        Dictionary with model, labels, linkage matrix, and metrics
    """
    # Compute linkage matrix
    linkage_matrix = linkage(X, method=method)
    
    # If n_clusters not specified, use a reasonable default
    if n_clusters is None:
        n_clusters = 4
    
    # Fit AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(X)
    
    # Compute metrics
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels)
    }
    
    return {
        'model': model,
        'labels': labels,
        'linkage_matrix': linkage_matrix,
        'n_clusters': n_clusters,
        'metrics': metrics
    }


def plot_dendrogram(linkage_matrix: np.ndarray, max_display: int = 30,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot hierarchical clustering dendrogram.
    
    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering
        max_display: Maximum number of clusters to display
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dendrogram(linkage_matrix, ax=ax, truncate_mode='lastp', p=max_display,
               show_leaf_counts=True, leaf_font_size=10)
    
    ax.set_xlabel('Sample Index or (Cluster Size)', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_clustering_methods(X: np.ndarray, kmeans_labels: np.ndarray,
                              hierarchical_labels: np.ndarray) -> pd.DataFrame:
    """
    Compare K-Means and Hierarchical clustering results.
    
    Args:
        X: Feature matrix
        kmeans_labels: Labels from K-Means
        hierarchical_labels: Labels from Hierarchical clustering
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison = pd.DataFrame({
        'Method': ['K-Means', 'Hierarchical'],
        'Silhouette Score': [
            silhouette_score(X, kmeans_labels),
            silhouette_score(X, hierarchical_labels)
        ],
        'Calinski-Harabasz Index': [
            calinski_harabasz_score(X, kmeans_labels),
            calinski_harabasz_score(X, hierarchical_labels)
        ],
        'Davies-Bouldin Index': [
            davies_bouldin_score(X, kmeans_labels),
            davies_bouldin_score(X, hierarchical_labels)
        ]
    })
    
    return comparison
