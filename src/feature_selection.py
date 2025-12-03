"""
Feature selection methods to identify important principal components.

This module implements Mutual Information, Lasso, and RFE for feature selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def mutual_info_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                         n_features: int = 20, random_state: int = 42) -> Dict:
    """
    Select features using Mutual Information.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        n_features: Number of features to select
        random_state: Random seed
        
    Returns:
        Dictionary with MI scores, selected features, and rankings
    """
    # Compute mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    
    # Create DataFrame with scores
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Select top features
    selected_features = mi_df.head(n_features)['feature'].tolist()
    selected_indices = [feature_names.index(f) for f in selected_features]
    
    return {
        'mi_scores': mi_scores,
        'mi_df': mi_df,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'n_features': n_features
    }


def lasso_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                   alpha: float = 0.01, random_state: int = 42) -> Dict:
    """
    Select features using Lasso (L1 regularization).
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        alpha: Regularization strength
        random_state: Random seed
        
    Returns:
        Dictionary with Lasso model, coefficients, and selected features
    """
    # Use LogisticRegression with L1 penalty for classification
    model = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear',
                               random_state=random_state, max_iter=1000)
    model.fit(X, y)
    
    # Get coefficients
    if len(model.coef_.shape) > 1:
        # Multi-class: take mean absolute coefficient across classes
        coefficients = np.mean(np.abs(model.coef_), axis=0)
    else:
        coefficients = np.abs(model.coef_[0])
    
    # Create DataFrame
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False)
    
    # Select non-zero coefficients
    selected_features = coef_df[coef_df['coefficient'] > 0]['feature'].tolist()
    selected_indices = [feature_names.index(f) for f in selected_features]
    
    return {
        'model': model,
        'coefficients': coefficients,
        'coef_df': coef_df,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'n_features': len(selected_features)
    }


def rfe_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                 n_features: int = 20, random_state: int = 42) -> Dict:
    """
    Select features using Recursive Feature Elimination.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        n_features: Number of features to select
        random_state: Random seed
        
    Returns:
        Dictionary with RFE model, rankings, and selected features
    """
    # Use Random Forest as base estimator
    estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    
    # Get rankings
    rankings_df = pd.DataFrame({
        'feature': feature_names,
        'ranking': rfe.ranking_,
        'selected': rfe.support_
    }).sort_values('ranking')
    
    selected_features = rankings_df[rankings_df['selected']]['feature'].tolist()
    selected_indices = [feature_names.index(f) for f in selected_features]
    
    return {
        'model': rfe,
        'rankings': rfe.ranking_,
        'rankings_df': rankings_df,
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'n_features': n_features
    }


def plot_feature_importance(importance_scores: np.ndarray, feature_names: List[str],
                           method_name: str, top_n: int = 20,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Args:
        importance_scores: Array of importance scores
        feature_names: List of feature names
        method_name: Name of feature selection method
        top_n: Number of top features to display
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Create DataFrame and sort
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(df)), df['importance'], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Features - {method_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['importance'])):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_feature_selection_methods(mi_results: Dict, lasso_results: Dict,
                                     rfe_results: Dict) -> pd.DataFrame:
    """
    Compare different feature selection methods.
    
    Args:
        mi_results: Results from Mutual Information
        lasso_results: Results from Lasso
        rfe_results: Results from RFE
        
    Returns:
        DataFrame with comparison statistics
    """
    mi_features = set(mi_results['selected_features'])
    lasso_features = set(lasso_results['selected_features'])
    rfe_features = set(rfe_results['selected_features'])
    
    # Find common features
    all_three = mi_features & lasso_features & rfe_features
    mi_lasso = mi_features & lasso_features
    mi_rfe = mi_features & rfe_features
    lasso_rfe = lasso_features & rfe_features
    
    comparison = pd.DataFrame({
        'Method': ['Mutual Information', 'Lasso', 'RFE'],
        'N Features Selected': [
            len(mi_features),
            len(lasso_features),
            len(rfe_features)
        ],
        'Common with All': [
            len(all_three),
            len(all_three),
            len(all_three)
        ]
    })
    
    return comparison


def evaluate_feature_subset(X_full: np.ndarray, X_reduced: np.ndarray,
                           y: np.ndarray, random_state: int = 42) -> Dict:
    """
    Evaluate model performance with full vs reduced feature set.
    
    Args:
        X_full: Full feature matrix
        X_reduced: Reduced feature matrix
        y: Target labels
        random_state: Random seed
        
    Returns:
        Dictionary with performance comparison
    """
    from sklearn.model_selection import cross_val_score
    from time import time
    
    # Use Random Forest for evaluation
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # Evaluate full features
    start = time()
    scores_full = cross_val_score(model, X_full, y, cv=5, scoring='accuracy')
    time_full = time() - start
    
    # Evaluate reduced features
    start = time()
    scores_reduced = cross_val_score(model, X_reduced, y, cv=5, scoring='accuracy')
    time_reduced = time() - start
    
    return {
        'full_features': {
            'n_features': X_full.shape[1],
            'cv_scores': scores_full,
            'mean_accuracy': np.mean(scores_full),
            'std_accuracy': np.std(scores_full),
            'time': time_full
        },
        'reduced_features': {
            'n_features': X_reduced.shape[1],
            'cv_scores': scores_reduced,
            'mean_accuracy': np.mean(scores_reduced),
            'std_accuracy': np.std(scores_reduced),
            'time': time_reduced
        },
        'speedup': time_full / time_reduced if time_reduced > 0 else 0
    }
