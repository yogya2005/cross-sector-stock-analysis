"""
Classification algorithms for predicting market regimes.

This module implements Random Forest, SVM, and k-NN classifiers with
evaluation metrics and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       n_estimators: int = 100, max_depth: Optional[int] = None,
                       random_state: int = 42, **kwargs) -> Dict:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        Dictionary with trained model and feature importances
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, **kwargs)
    model.fit(X_train, y_train)
    
    return {
        'model': model,
        'feature_importances': model.feature_importances_,
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
             kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
             random_state: int = 42, **kwargs) -> Dict:
    """
    Train SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        kernel: Kernel type
        C: Regularization parameter
        gamma: Kernel coefficient
        random_state: Random seed
        **kwargs: Additional parameters for SVC
        
    Returns:
        Dictionary with trained model
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state,
                probability=True, **kwargs)
    model.fit(X_train, y_train)
    
    return {
        'model': model,
        'kernel': kernel,
        'C': C,
        'gamma': gamma
    }


def train_knn(X_train: np.ndarray, y_train: np.ndarray,
             n_neighbors: int = 5, weights: str = 'uniform',
             metric: str = 'euclidean', **kwargs) -> Dict:
    """
    Train k-NN classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_neighbors: Number of neighbors
        weights: Weight function
        metric: Distance metric
        **kwargs: Additional parameters for KNeighborsClassifier
        
    Returns:
        Dictionary with trained model
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                                metric=metric, **kwargs)
    model.fit(X_train, y_train)
    
    return {
        'model': model,
        'n_neighbors': n_neighbors,
        'weights': weights,
        'metric': metric
    }


def evaluate_classifier(model, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       cv_folds: int = 5) -> Dict:
    """
    Evaluate classifier with multiple metrics.
    
    Args:
        model: Trained classifier
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    
    # Metrics
    n_classes = len(np.unique(y_train))
    average = 'binary' if n_classes == 2 else 'weighted'
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'precision': precision_score(y_test, y_test_pred, average=average, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, average=average, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }
    
    # Add probabilities if available
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)
        metrics['y_test_proba'] = y_test_proba
        
        # AUC-ROC for multi-class
        if n_classes > 2:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
            try:
                metrics['auc_roc'] = roc_auc_score(y_test_bin, y_test_proba, average='weighted')
            except:
                metrics['auc_roc'] = None
        else:
            metrics['auc_roc'] = roc_auc_score(y_test, y_test_proba[:, 1])
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                         title: str = 'Confusion Matrix',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves(y_test: np.ndarray, y_test_proba: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   title: str = 'ROC Curves',
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_test: True labels
        y_test_proba: Predicted probabilities
        class_names: Names of classes
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_classes = y_test_proba.shape[1]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    # Binarize labels
    unique_classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=unique_classes)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        if i < y_test_bin.shape[1]:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
            roc_auc = roc_auc_score(y_test_bin[:, i], y_test_proba[:, i])
            
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def hyperparameter_tuning_rf(X_train: np.ndarray, y_train: np.ndarray,
                             param_grid: Optional[Dict] = None,
                             cv: int = 5, random_state: int = 42) -> Dict:
    """
    Perform hyperparameter tuning for Random Forest.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary with best model and parameters
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    rf = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'grid_search': grid_search,
        'cv_results': grid_search.cv_results_
    }


def hyperparameter_tuning_svm(X_train: np.ndarray, y_train: np.ndarray,
                              param_grid: Optional[Dict] = None,
                              cv: int = 5, random_state: int = 42) -> Dict:
    """
    Perform hyperparameter tuning for SVM.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Dictionary with best model and parameters
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly']
        }
    
    svm = SVC(random_state=random_state, probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'grid_search': grid_search,
        'cv_results': grid_search.cv_results_
    }


def hyperparameter_tuning_knn(X_train: np.ndarray, y_train: np.ndarray,
                              param_grid: Optional[Dict] = None,
                              cv: int = 5) -> Dict:
    """
    Perform hyperparameter tuning for k-NN.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with best model and parameters
    """
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'grid_search': grid_search,
        'cv_results': grid_search.cv_results_
    }


def compare_models(models_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple classifiers.
    
    Args:
        models_dict: Dictionary with model names as keys and evaluation results as values
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison = []
    
    for name, metrics in models_dict.items():
        comparison.append({
            'Model': name,
            'Train Accuracy': metrics.get('train_accuracy', 0),
            'Test Accuracy': metrics.get('test_accuracy', 0),
            'CV Mean': metrics.get('cv_mean', 0),
            'CV Std': metrics.get('cv_std', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'AUC-ROC': metrics.get('auc_roc', 0)
        })
    
    return pd.DataFrame(comparison)
